
import time
import gc
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from ptflops import get_model_complexity_info
import copy
from src.models.bce_that import THAT
from src.models.losses.supervised_loss import JointDistillationLoss
from src.utils import *
import wandb
import torch
from torch.optim.lr_scheduler import LambdaLR
import math
from configs.preset import preset
import torch.nn.functional as F


torch.set_float32_matmul_precision("high")
torch._dynamo.config.cache_size_limit = 65536
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)

class MultiSenseX(torch.nn.Module):
    def __init__(self,
                 var_x_shape,
                 embedding_dim=100,
                 threshold=0.5,
                 location_only=False):

        super().__init__()
        self.backbone = THAT(var_x_shape, [embedding_dim])


        # Location head (multilabel classification)
        self.location_head = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 5)
        )

        # Activity heads (one per location). Only built for the joint model.
        self.location_only = location_only
        if not location_only:
            self.act_heads = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(embedding_dim, 32),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(32, 9)
                ) for _ in range(5)
            ])

        self.sigmoid = torch.nn.Sigmoid()
        self.threshold = threshold  # For activating locations

    def forward(self, x):
        """
        Args:
            z: Backbone features of shape (batch_size, var_dim_in)
        Returns:
            act_logits: Activity logits (batch_size, 5, 9), or None for the location-only model
            loc_pred: Location probabilities (batch_size, 5)
            mask: Boolean mask of locations above the threshold (batch_size, 5)
        """
        z = self.backbone(x)

        # --- Location Prediction ---
        logit_loc = self.location_head(z)
        loc_pred = self.sigmoid(logit_loc)  # Shape: (batch_size, 5)

        # --- Mask for Active Locations ---
        mask = loc_pred > self.threshold  # (batch_size, 5)

        if self.location_only:
            return None, loc_pred, mask

        # --- Activity Prediction for Active Locations ---
        act_logits = torch.stack([head(z) for head in self.act_heads], dim=1) #(batch_size, 5, 9)

        return act_logits, loc_pred, mask

class JointActLocDataset(torch.utils.data.Dataset):
    def __init__(self, CSI, y_act, y_loc):
        self.X = CSI
        self.y_loc = y_loc
        self.y_act = y_act

    def __len__(self)  -> int :
        return self.X.shape[0]
    def __getitem__(self, idx) -> tuple:
        return self.X[idx], self.y_act[idx], self.y_loc[idx]

#
##
def run_multi_senseX_joint(data_train_x,
                    data_train_y_loc,
                    data_train_y_act,
                    data_test_x,
                    data_test_y_loc,
                    data_test_y_act,
                     var_repeat=10):
    """
    [description]
    : run WiFi-based model THAT_ENCODER
    [parameter]
    : data_train_x: numpy array, CSI amplitude to train model
    : data_train_y: numpy array, labels to train model
    : data_test_x: numpy array, CSI amplitude to test model
    : data_test_y: numpy array, labels to test model
    : var_repeat: int, number of repeated experiments
    [return]
    : result: dict, results of experiments
    """
    #
    ##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    ##
    ## ============================================ Preprocess ============================================
    #
    ##
    (data_valid_x, data_test_x,
     data_valid_y_loc, data_test_y_loc,
     data_valid_y_act, data_test_y_act) = my_train_test_split(data_test_x, data_test_y_loc, data_test_y_act,
                                                              test_size=0.5, random_state=103)

    data_valid_x = data_valid_x.reshape(data_valid_x.shape[0], data_valid_x.shape[1], -1)
    data_train_x = data_train_x.reshape(data_train_x.shape[0], data_train_x.shape[1], -1)
    data_test_x = data_test_x.reshape(data_test_x.shape[0], data_test_x.shape[1], -1)
    #
    ## shape for model
    var_x_shape, var_y_shape_loc, var_y_shape_act = data_train_x[0].shape, data_train_y_loc.shape[
                                                                           1:], data_train_y_act.shape[1:]
    #
    data_train_set = JointActLocDataset(data_train_x, data_train_y_act, data_train_y_loc)

    data_valid_set = JointActLocDataset(data_valid_x, data_valid_y_act, data_valid_y_loc)

    #
    ##
    ## ========================================= Train & Evaluate =========================================
    #
    ##
    result_ppp_act = []
    result_total_error_act = []
    result_precision_act = []
    result_recall_act = []
    result_f1_score_act = []
    result_avg_count_error_act = []

    # Store location results
    result_ppp_loc = []
    result_total_error_loc = []
    result_precision_loc = []
    result_recall_loc = []
    result_f1_score_loc = []
    result_avg_count_error_loc = []

    # Store timing results
    result_time_train = []
    result_time_test = []

    #
    var_macs, var_params = get_model_complexity_info(MultiSenseX(var_x_shape),
                                                     var_x_shape, as_strings=False)
    #
    print("Parameters:", var_params, "- FLOPs:", var_macs * 2)
    #
    ##
    for var_r in range(var_repeat):
        #
        ##
        print("Repeat", var_r)
        name_run = f"MultiSenseX{var_r}_" + "_".join(preset["data"]["environment"])

        run = wandb.init(
            project="multiSenseX",
            name= name_run,
            config=preset,
            reinit=True  # Allow multiple wandb.init() calls in the same process
        )
        #
        torch.random.manual_seed(var_r + 39)
        #
        model_multiSenseX = MultiSenseX(var_x_shape,
                 embedding_dim=100,
                 threshold=0.5).to(device)
        #

        optimizer = torch.optim.Adam(model_multiSenseX.parameters(),
                                         lr=preset["nn"]["lr"],
                                         weight_decay=preset["nn"]["weight_decay"])

        #
        loss_mode = "multi_senseX"
        var_time_0 = time.time()
        #
        ## ---------------------------------------- Train -----------------------------------------
        #
        var_best_weight = train(model = model_multiSenseX,
                                optimizer = optimizer,
                                data_train_set = data_train_set,
                                data_test_set = data_valid_set,
                                var_threshold = preset["nn"]["threshold"],
                                var_batch_size = preset["nn"]["batch_size"],
                                var_epochs = preset["nn"]["epoch"],
                                device = device,
                                var_mode = loss_mode)
        #
        var_time_1 = time.time()

        ##
        ## ---------------------------------------- Test ------------------------------------------
        #
        model_multiSenseX.load_state_dict(var_best_weight)
        #
        with torch.no_grad():
            predict_test_y_act, predict_test_y_loc, mask = model_multiSenseX(torch.from_numpy(data_test_x).to(device))
        #
        # predict_test_y = torch.clamp(torch.round(predict_test_y), min=0, max=5).float()
        predict_test_act = predict_test_y_act.detach().cpu().numpy()
        predict_test_loc = predict_test_y_loc.detach().cpu().numpy()

        #
        var_time_2 = time.time()
        #
        ## -------------------------------------- Evaluate ----------------------------------------
        #
        ##

        dict_true_acc_act, dict_true_acc_loc = performance_metrics_joint_multiSensX(data_test_y_act, predict_test_act,
                                                                         data_test_y_loc, predict_test_loc)

        wandb.log({
            "repeat": var_r,
            "train_time": var_time_1 - var_time_0,
            "test_time": var_time_2 - var_time_1,

            # Activity metrics
            "ACT_TOTAL_TESTSET_ERROR": dict_true_acc_act['total_error'],
            "ACT_TOTAL_TESTSET_perfect_prediction_percentage": dict_true_acc_act['perfect_prediction_percentage'],
            "ACT_TOTAL_ACCURACY": dict_true_acc_act['accuracy'],
            "ACT_mean_count_error": dict_true_acc_act['mean_count_error'],
            "ACT_error_per_person_1": dict_true_acc_act['error_per_person'][0],
            "ACT_error_per_person_2": dict_true_acc_act['error_per_person'][1],
            "ACT_error_per_person_3": dict_true_acc_act['error_per_person'][2],
            "ACT_error_per_person_4": dict_true_acc_act['error_per_person'][3],
            "ACT_error_per_person_5": dict_true_acc_act['error_per_person'][4],
            "ACT_precision": dict_true_acc_act['precision'],
            "ACT_recall": dict_true_acc_act['recall'],
            "ACT_f1_score": dict_true_acc_act['f1_score'],

            # Location metrics
            "LOC_TOTAL_TESTSET_ERROR": dict_true_acc_loc['total_error'],
            "LOC_TOTAL_TESTSET_perfect_prediction_percentage": dict_true_acc_loc['perfect_prediction_percentage'],
            "LOC_TOTAL_ACCURACY": dict_true_acc_loc['accuracy'],
            "LOC_mean_count_error": dict_true_acc_loc['mean_count_error'],
            "LOC_error_per_person_1": dict_true_acc_loc['error_per_person'][0],
            "LOC_error_per_person_2": dict_true_acc_loc['error_per_person'][1],
            "LOC_error_per_person_3": dict_true_acc_loc['error_per_person'][2],
            "LOC_error_per_person_4": dict_true_acc_loc['error_per_person'][3],
            "LOC_error_per_person_5": dict_true_acc_loc['error_per_person'][4],
            "LOC_precision": dict_true_acc_loc['precision'],
            "LOC_recall": dict_true_acc_loc['recall'],
            "LOC_f1_score": dict_true_acc_loc['f1_score']
        })
        #
        #

        #
        result_ppp_act.append(dict_true_acc_act['perfect_prediction_percentage'])
        result_total_error_act.append(dict_true_acc_act['total_error'])
        result_precision_act.append(dict_true_acc_act['precision'])
        result_recall_act.append(dict_true_acc_act['recall'])
        result_f1_score_act.append(dict_true_acc_act['f1_score'])
        result_avg_count_error_act.append(dict_true_acc_act['mean_count_error'])

        result_ppp_loc.append(dict_true_acc_loc['perfect_prediction_percentage'])
        result_total_error_loc.append(dict_true_acc_loc['total_error'])
        result_precision_loc.append(dict_true_acc_loc['precision'])
        result_recall_loc.append(dict_true_acc_loc['recall'])
        result_f1_score_loc.append(dict_true_acc_loc['f1_score'])
        result_avg_count_error_loc.append(dict_true_acc_loc['mean_count_error'])

    wandb.log({
        # Activity averages
        "ACT_avg_accuracy": sum(result_ppp_act) / len(result_ppp_act),
        "ACT_avg_total_error": sum(result_total_error_act) / len(result_total_error_act),
        "ACT_avg_precision": sum(result_precision_act) / len(result_precision_act),
        "ACT_avg_recall": sum(result_recall_act) / len(result_recall_act),
        "ACT_avg_f1_score": sum(result_f1_score_act) / len(result_f1_score_act),
        "ACT_avg_count_error": sum(result_avg_count_error_act) / len(result_avg_count_error_act),

        # Location averages
        "LOC_avg_accuracy": sum(result_ppp_loc) / len(result_ppp_loc),
        "LOC_avg_total_error": sum(result_total_error_loc) / len(result_total_error_loc),
        "LOC_avg_precision": sum(result_precision_loc) / len(result_precision_loc),
        "LOC_avg_recall": sum(result_recall_loc) / len(result_recall_loc),
        "LOC_avg_f1_score": sum(result_f1_score_loc) / len(result_f1_score_loc),
        "LOC_avg_count_error": sum(result_avg_count_error_loc) / len(result_avg_count_error_loc),
    })

    # viz_stats = visualize_model_performance(
    #     y_pred=predict_test_y,
    #     y_true=data_test_y_act,
    #     var_mode=var_mode,
    #     save_dir=f'./visualizations/experiment_{var_r}_{var_mode}'
    # )
    # print("\nDetailed Performance Analysis:")
    # print(f"Mean Error: {viz_stats['mean_error']:.4f} ± {viz_stats['error_std']:.4f}")
    # print("\nClass-wise Mean Absolute Error:")
    # for i, error in enumerate(viz_stats['class_wise_mae']):
    #     print(f"Class {i}: {error:.4f}")
    # print(f"\nPerfect Predictions: {viz_stats['perfect_predictions'] * 100:.2f}%")
    wandb.finish()
    return dict_true_acc_act, dict_true_acc_loc

def  multisense_loss(act_logits, activity_targets, loc_pred, location_targets,   mask):
    """
    Calculate the combined loss for location and activity prediction.

    Args:
        act_logits: Activity logits of shape (batch_size, 5, 9)
        loc_pred: Location predictions (sigmoid probabilities) of shape (batch_size, 5)
        location_targets: Ground truth location labels of shape (batch_size, 5)
        activity_targets: Ground truth activity targets of shape (batch_size, 5, 9)
        mask: Boolean mask of shape (batch_size, 5) indicating locations with predicted people
    """

    # Location loss - Binary Cross Entropy for multi-label classification
    location_loss = F.binary_cross_entropy(loc_pred, location_targets)

    # Convert activity_targets to class indices if they're one-hot encoded
    if activity_targets.dim() == 3 and activity_targets.size(2) == 9:
        activity_indices = torch.argmax(activity_targets, dim=2)  # (batch_size, 5)
    else:
        activity_indices = activity_targets  # Already indices

    # Get ground truth mask (where there's actually a person)
    gt_mask = (location_targets > 0.5)

    # Combine prediction mask with ground truth mask
        # valid_mask = mask & gt_mask

    valid_mask = mask | ~mask

    # If there are any valid locations, compute activity loss
    if valid_mask.any():
        # Reshape logits and indices for loss calculation
        b, l, c = act_logits.shape  # batch, locations, classes

        # Creating a mask that keeps the same shape for proper element selection
        expanded_mask = valid_mask.unsqueeze(-1).expand_as(act_logits)

        # Compute loss only for valid locations
        act_flat = act_logits.reshape(-1, c)
        indices_flat = activity_indices.reshape(-1)
        mask_flat = valid_mask.reshape(-1)

        # Select only entries where mask is True
        act_masked = act_flat[mask_flat]
        indices_masked = indices_flat[mask_flat]

        # Compute cross-entropy only on valid entries
        activity_loss = F.cross_entropy(act_masked, indices_masked)
    else:
        # No valid locations
        activity_loss = torch.tensor(0.0, device=loc_pred.device)

    total_loss = location_loss + activity_loss

    return total_loss, activity_loss, location_loss

def train(model,
          optimizer,
          data_train_set: TensorDataset,
          data_test_set: TensorDataset,
          var_threshold: float,
          var_batch_size: int,
          var_epochs: int,
          device,
          var_mode: str,
          patience: int = 150,  # Added patience parameter
          var_selection: str = "f1_then_ppp",
          teacher=None,
          kd_loss=None,
          kd_weight: float = 0.0):

    data_train_loader = torch.utils.data.DataLoader(data_train_set, var_batch_size, shuffle=True, pin_memory=True)
    data_test_loader =  torch.utils.data.DataLoader(data_test_set, len(data_test_set))

    # Initialize early stopping variables
    var_best_f1_score_act = 0
    var_best_PPP_act = 0
    var_best_total_error_act = np.inf
    var_best_f1_score_loc = 0
    var_best_PPP_loc = 0
    var_best_weight = None
    counter = 0  # Counter for patience

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=preset["nn"]["scheduler"]["num_warmup_epochs"] * len(data_train_loader),
        num_training_steps=preset["nn"]["epoch"] * len(data_train_loader),
        min_lr_ratio=preset["nn"]["scheduler"]["min_lr_ratio"]
    )

    def apply_augmentation(x_batch):
        noise = torch.randn_like(x_batch) * 0.1
        x_batch = x_batch + noise
        scale = torch.rand(x_batch.size(0), 1, device=x_batch.device) * 0.2 + 0.9
        x_batch = x_batch * scale.unsqueeze(-1)
        mask = torch.bernoulli(torch.ones_like(x_batch) * 0.96)
        x_batch = x_batch * mask

        return x_batch

    for var_epoch in range(var_epochs):
        var_time_e0 = time.time()
        model.train()
        total_batches = len(data_train_loader)

        for batch_idx, data_batch in enumerate(data_train_loader):

            data_batch_x, data_batch_y_act, data_batch_y_loc = data_batch
            data_batch_x = data_batch_x.to(device)
            data_batch_y_act = data_batch_y_act.to(device)
            data_batch_y_loc = data_batch_y_loc.to(device)

            if model.training:
                data_batch_x = apply_augmentation(data_batch_x)

            predict_train_y_act, predict_train_y_loc, mask = model(data_batch_x)

            var_loss_train, _, _ = multisense_loss(predict_train_y_act, data_batch_y_act,
                                  predict_train_y_loc,  data_batch_y_loc.float(), mask)

            if teacher is not None and kd_loss is not None and kd_weight > 0:
                with torch.no_grad():
                    teacher_y_act, teacher_y_loc, _ = teacher(data_batch_x)
                var_loss_train = var_loss_train + kd_weight * kd_loss(
                    (predict_train_y_act, predict_train_y_loc),
                    (teacher_y_act, teacher_y_loc))

            optimizer.zero_grad()
            var_loss_train.backward()
            optimizer.step()
            scheduler.step()

        data_batch_y_act = data_batch_y_act.detach().cpu().numpy()
        data_batch_y_loc = data_batch_y_loc.detach().cpu().numpy()

        predict_train_y_act = predict_train_y_act.detach().cpu().numpy()
        predict_train_y_loc = predict_train_y_loc.detach().cpu().numpy()

        # Calculate performance metrics for training
        dict_error_train_act, dict_error_train_loc = performance_metrics_joint_multiSensX(
            y_true_act=data_batch_y_act,
            y_pred_act=predict_train_y_act,
            y_true_loc=data_batch_y_loc,
            y_pred_loc=predict_train_y_loc,
        )
        # dict_error_train_act = performance_metrics(data_batch_y_act, predict_train_y_act,
        #                                            var_mode=var_mode, var_threshold=var_threshold)
        # dict_error_train_loc = performance_metrics(data_batch_y_loc, predict_train_y_loc,
        #                                            var_mode=var_mode, var_threshold=var_threshold)

        model.eval()
        with torch.no_grad():
            data_test_x, data_test_y_act, data_test_y_loc = next(iter(data_test_loader))
            data_test_x = data_test_x.to(device)
            data_test_y_act = data_test_y_act.to(device)
            data_test_y_loc = data_test_y_loc.to(device)
            predict_test_y_act, predict_test_y_loc, mask= model(data_test_x)
            var_loss_test, _, _ = multisense_loss(predict_test_y_act, data_test_y_act.float(),
                                 predict_test_y_loc, data_test_y_loc.float(), mask)

            # Convert to numpy for metrics calculation
            data_test_y_act = data_test_y_act.detach().cpu().numpy()
            data_test_y_loc = data_test_y_loc.detach().cpu().numpy()
            predict_test_y_act = predict_test_y_act.detach().cpu().numpy()
            predict_test_y_loc = predict_test_y_loc.detach().cpu().numpy()

            # # Calculate performance metrics for both activity and location
            # dict_error_test_act = performance_metrics(data_test_y_act, predict_test_y_act,
            #                                           var_mode, var_threshold)
            # dict_error_test_loc = performance_metrics(data_test_y_loc, predict_test_y_loc,
            #                                           var_mode, var_threshold)

            dict_error_test_act, dict_error_test_loc = performance_metrics_joint_multiSensX(
                y_true_act=data_test_y_act,
                y_pred_act=predict_test_y_act,
                y_true_loc=data_test_y_loc,
                y_pred_loc=predict_test_y_loc
            )
        # Log metrics for both activity and location
        wandb.log({
            "epoch": var_epoch,
            "train_loss": var_loss_train.item(),
            "test_loss": var_loss_test.item(),

            # Activity metrics
            "ACT_total_error_train": dict_error_train_act['total_error'],
            "ACT_total_error_test": dict_error_test_act['total_error'],
            "ACT_perfect_prediction_percentage_test": dict_error_test_act['perfect_prediction_percentage'],
            "ACT_perfect_prediction_percentage_train": dict_error_train_act['perfect_prediction_percentage'],
            "ACT_accuracy_test": dict_error_test_act['accuracy'],
            "ACT_accuracy_train": dict_error_train_act['accuracy'],
            "ACT_precision": dict_error_test_act['precision'],
            "ACT_recall": dict_error_test_act['recall'],
            "ACT_f1_score": dict_error_test_act['f1_score'],

            # Location metrics
            "LOC_total_error_train": dict_error_train_loc['total_error'],
            "LOC_total_error_test": dict_error_test_loc['total_error'],
            "LOC_perfect_prediction_percentage_test": dict_error_test_loc['perfect_prediction_percentage'],
            "LOC_perfect_prediction_percentage_train": dict_error_train_loc['perfect_prediction_percentage'],
            "LOC_accuracy_test": dict_error_test_loc['accuracy'],
            "LOC_accuracy_train": dict_error_train_loc['accuracy'],
            "LOC_precision": dict_error_test_loc['precision'],
            "LOC_recall": dict_error_test_loc['recall'],
            "LOC_f1_score": dict_error_test_loc['f1_score'],

            # Other metrics
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # Print training progress
        print(f"Epoch {var_epoch}/{var_epochs}",
              "- %.6fs" % (time.time() - var_time_e0),
              "- Loss %.6f" % var_loss_train.cpu(),
              "- Test Loss %.6f" % var_loss_test.cpu())

        print("ACTIVITY:",
              "- Total Error Train %.6f" % dict_error_train_act['total_error'],
              "- Total Error Test %.6f" % dict_error_test_act['total_error'],
              "- PPP Train %.6f" % dict_error_train_act['perfect_prediction_percentage'],
              "- PPP Test %.6f" % dict_error_test_act['perfect_prediction_percentage'],
              "- Acc Train %.6f" % dict_error_train_act['accuracy'],
              "- Acc Test %.6f" % dict_error_test_act['accuracy'],
              "- Precision %.6f" % dict_error_test_act['precision'],
              "- Recall %.6f" % dict_error_test_act['recall'],
              "- F1 Score %.6f" % dict_error_test_act['f1_score'])

        print("LOCATION:",
              "- Total Error Train %.6f" % dict_error_train_loc['total_error'],
              "- Total Error Test %.6f" % dict_error_test_loc['total_error'],
              "- PPP Train %.6f" % dict_error_train_loc['perfect_prediction_percentage'],
              "- PPP Test %.6f" % dict_error_test_loc['perfect_prediction_percentage'],
              "- Acc Train %.6f" % dict_error_train_loc['accuracy'],
              "- Acc Test %.6f" % dict_error_test_loc['accuracy'],
              "- Precision %.6f" % dict_error_test_loc['precision'],
              "- Recall %.6f" % dict_error_test_loc['recall'],
              "- F1 Score %.6f" % dict_error_test_loc['f1_score'])

        if var_selection == "ppp_then_error":
            improved = (
                dict_error_test_act['perfect_prediction_percentage'] > var_best_PPP_act
                or (dict_error_test_act['perfect_prediction_percentage'] == var_best_PPP_act
                    and dict_error_test_act['total_error'] < var_best_total_error_act)
            )
        else:
            improved = (dict_error_test_act['f1_score'] > var_best_f1_score_act and
                        dict_error_test_act['perfect_prediction_percentage'] > var_best_PPP_act)

        if improved:

            # Update best scores
            var_best_PPP_act = dict_error_test_act['perfect_prediction_percentage']
            var_best_f1_score_act = dict_error_test_act['f1_score']
            var_best_total_error_act = dict_error_test_act['total_error']

            # Still track location metrics, but don't use them for model selection
            var_best_PPP_loc = dict_error_test_loc['perfect_prediction_percentage']
            var_best_f1_score_loc = dict_error_test_loc['f1_score']

            var_best_weight = copy.deepcopy(model.state_dict())
            var_epoch_saved = var_epoch
            counter = 0  # Reset counter
        else:
            counter += 1  # Increment counter

        # Early stopping check
        if counter >= patience:
            print(f"Early stopping triggered at epoch {var_epoch}")
            break

    if var_best_weight is None:
        # No epoch beat the initial 0-baseline; keep the last epoch rather than returning None.
        var_best_weight = copy.deepcopy(model.state_dict())
        var_epoch_saved = var_epoch

    print(f"Epoch that the model was saved {var_epoch_saved}")
    print(f"Best activity metrics - F1: {var_best_f1_score_act:.6f}, PPP: {var_best_PPP_act:.6f}")
    print(f"Best location metrics - F1: {var_best_f1_score_loc:.6f}, PPP: {var_best_PPP_loc:.6f}")

    return var_best_weight


def train_location(model,
                   optimizer,
                   data_train_set: TensorDataset,
                   data_test_set: TensorDataset,
                   var_batch_size: int,
                   var_epochs: int,
                   device,
                   patience: int = 150):
    """
    [description]
    : train the location-only MultiSenseX head. The objective is the binary cross-entropy of the
      5 multilabel location outputs; model selection is by perfect-prediction percentage, then by
      total error.
    : data_train_set / data_test_set: TensorDataset of (CSI, location_target), location_target is
      (N, 5) with 0/1 entries.
    """
    data_train_loader = torch.utils.data.DataLoader(data_train_set, var_batch_size, shuffle=True, pin_memory=True)
    data_test_loader = torch.utils.data.DataLoader(data_test_set, len(data_test_set))

    var_best_PPP = 0
    var_best_total_error = np.inf
    var_best_weight = None
    var_epoch_saved = 0
    counter = 0  # Counter for patience

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=preset["nn"]["scheduler"]["num_warmup_epochs"] * len(data_train_loader),
        num_training_steps=preset["nn"]["epoch"] * len(data_train_loader),
        min_lr_ratio=preset["nn"]["scheduler"]["min_lr_ratio"]
    )

    def apply_augmentation(x_batch):
        noise = torch.randn_like(x_batch) * 0.1
        x_batch = x_batch + noise
        scale = torch.rand(x_batch.size(0), 1, device=x_batch.device) * 0.2 + 0.9
        x_batch = x_batch * scale.unsqueeze(-1)
        mask = torch.bernoulli(torch.ones_like(x_batch) * 0.96)
        x_batch = x_batch * mask

        return x_batch

    for var_epoch in range(var_epochs):
        var_time_e0 = time.time()
        model.train()
        for data_batch_x, data_batch_y_loc in data_train_loader:
            data_batch_x = data_batch_x.to(device)
            data_batch_y_loc = data_batch_y_loc.to(device)

            data_batch_x = apply_augmentation(data_batch_x)

            _, predict_train_y_loc, _ = model(data_batch_x)
            var_loss_train = F.binary_cross_entropy(predict_train_y_loc, data_batch_y_loc.float())

            optimizer.zero_grad()
            var_loss_train.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            data_test_x, data_test_y_loc = next(iter(data_test_loader))
            _, predict_test_y_loc, _ = model(data_test_x.to(device))
            var_loss_test = F.binary_cross_entropy(predict_test_y_loc, data_test_y_loc.to(device).float())
            dict_error_test_loc = calculate_scores(
                data_test_y_loc.numpy(),
                (predict_test_y_loc.cpu().numpy() > 0.5).astype(int))

        wandb.log({
            "epoch": var_epoch,
            "train_loss": var_loss_train.item(),
            "test_loss": var_loss_test.item(),
            "LOC_total_error_test": dict_error_test_loc['total_error'],
            "LOC_perfect_prediction_percentage_test": dict_error_test_loc['perfect_prediction_percentage'],
            "LOC_accuracy_test": dict_error_test_loc['accuracy'],
            "LOC_precision": dict_error_test_loc['precision'],
            "LOC_recall": dict_error_test_loc['recall'],
            "LOC_f1_score": dict_error_test_loc['f1_score'],
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        print(f"Epoch {var_epoch}/{var_epochs}",
              "- %.6fs" % (time.time() - var_time_e0),
              "- Loss %.6f" % var_loss_train.cpu(),
              "- Test Loss %.6f" % var_loss_test.cpu())
        print("LOCATION:",
              "- Total Error Test %.6f" % dict_error_test_loc['total_error'],
              "- PPP Test %.6f" % dict_error_test_loc['perfect_prediction_percentage'],
              "- Acc Test %.6f" % dict_error_test_loc['accuracy'],
              "- Precision %.6f" % dict_error_test_loc['precision'],
              "- Recall %.6f" % dict_error_test_loc['recall'],
              "- F1 Score %.6f" % dict_error_test_loc['f1_score'])

        if (dict_error_test_loc['perfect_prediction_percentage'] > var_best_PPP
                or (dict_error_test_loc['perfect_prediction_percentage'] == var_best_PPP
                    and dict_error_test_loc['total_error'] < var_best_total_error)):
            var_best_PPP = dict_error_test_loc['perfect_prediction_percentage']
            var_best_total_error = dict_error_test_loc['total_error']
            var_best_weight = copy.deepcopy(model.state_dict())
            var_epoch_saved = var_epoch
            counter = 0  # Reset counter
        else:
            counter += 1  # Increment counter

        # Early stopping check
        if counter >= patience:
            print(f"Early stopping triggered at epoch {var_epoch}")
            break

    if var_best_weight is None:
        # No epoch beat the initial 0-baseline; keep the last epoch rather than returning None.
        var_best_weight = copy.deepcopy(model.state_dict())

    print(f"Epoch that the model was saved {var_epoch_saved}")
    print(f"Best location metrics - PPP: {var_best_PPP:.6f}, Total Error: {var_best_total_error:.6f}")

    return var_best_weight


## ====================================================================================================================
# FEW-SHOT KNOWLEDGE DISTILLATION (MultiSenseX)

def _mean_std_se(arr):
    """Mean, sample std and standard error over the repeats."""
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    se = float(std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, std, se


def run_multi_senseX_few_shot(data_train_x,
                              data_train_y_act,
                              data_train_y_loc,
                              test_sets_by_env,
                              var_few_shot_ratio=0.05,
                              var_kd_weight=1.0,
                              var_kd_temperature=1.0,
                              var_teacher_epochs=None,
                              var_student_epochs=None,
                              var_repeat=10, var_env="empty_room",
                              save_path="./visualizations/temp"):
    """
    [description]
    : Few-shot knowledge distillation for MultiSenseX, trained on a single environment and tested on
      the others. A teacher MultiSenseX is trained on the full training environment, frozen, and used
      to supervise a student MultiSenseX that only sees a small fraction (var_few_shot_ratio) of that
      same environment. The student objective is
      multisense_loss + var_kd_weight * JointDistillationLoss against the teacher.
    [parameter]
    : data_train_x: numpy array, CSI amplitude of the single training environment
    : data_train_y_act: numpy array, activity labels (N, num_obj_queries, 9)
    : data_train_y_loc: numpy array, location labels (N, num_obj_queries)
    : test_sets_by_env: dict, {env_name: (X, y_act, y_loc)} test sets of the other environments
    : var_few_shot_ratio: float, fraction of the training environment used to train the student
    : var_kd_weight: float, weight of the distillation term
    : var_kd_temperature: float, temperature of the distillation soft targets
    : var_teacher_epochs: int, teacher training epochs (defaults to preset["nn"]["epoch"])
    : var_student_epochs: int, student training epochs (defaults to preset["nn"]["epoch"])
    : var_repeat: int, number of repeated experiments
    : var_env: str or list, training environment name(s) used for the run name
    : save_path: str, directory kept for symmetry with the AMAR few-shot runner (no figures written)
    [return]
    : all_envs_results: dict, {env_name: {"act": {...}, "loc": {...}}} averaged metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if var_teacher_epochs is None:
        var_teacher_epochs = preset["nn"]["epoch"]
    if var_student_epochs is None:
        var_student_epochs = preset["nn"]["epoch"]
    env_name = var_env if isinstance(var_env, str) else "_".join(var_env)

    #
    ## ============================================ Preprocess ============================================
    #
    data_train_x = data_train_x.reshape(data_train_x.shape[0], data_train_x.shape[1], -1)
    var_x_shape = data_train_x[0].shape

    ## NOTE: index-based partition rather than chained train_test_split calls. The raw CSI arrays are
    ## multi-GB per environment, and a chained split would transiently hold the original array plus a
    ## full-size "rest" copy plus the teacher copies.
    num_train = data_train_x.shape[0]
    num_few = max(1, int(round(var_few_shot_ratio * num_train)))
    if num_few >= num_train - 1:
        raise ValueError(
            f"var_few_shot_ratio={var_few_shot_ratio} leaves no validation data "
            f"({num_few}/{num_train} training-environment samples). Lower the ratio or provide more data."
        )
    shuffle_idx = np.random.RandomState(39).permutation(num_train)
    few_idx = shuffle_idx[:num_few]
    rest_idx = shuffle_idx[num_few:]
    num_valid = max(1, int(round(0.1 * rest_idx.shape[0])))
    valid_idx = rest_idx[:num_valid]
    teacher_idx = rest_idx[num_valid:]

    student_train_set = JointActLocDataset(data_train_x[few_idx], data_train_y_act[few_idx], data_train_y_loc[few_idx])
    teacher_valid_set = JointActLocDataset(data_train_x[valid_idx], data_train_y_act[valid_idx], data_train_y_loc[valid_idx])
    teacher_train_set = JointActLocDataset(data_train_x[teacher_idx], data_train_y_act[teacher_idx], data_train_y_loc[teacher_idx])
    print(f"Training environment [{env_name}] - teacher train {teacher_idx.shape[0]} | "
          f"student few-shot train {num_few}/{num_train} ({var_few_shot_ratio:.2%}) | "
          f"student validation {num_valid}")
    print(f"Test environments: {list(test_sets_by_env.keys())}")

    #
    ## ---------------------------------------- Complexity ----------------------------------------
    #
    var_macs, var_params = get_model_complexity_info(MultiSenseX(var_x_shape),
                                                     var_x_shape, as_strings=False)
    print("Parameters:", var_params, "- FLOPs:", var_macs * 2)

    #
    ## ========================================= Train & Evaluate =========================================
    #
    env_result = {}
    for var_r in range(var_repeat):
        #
        ##
        print("Repeat", var_r)
        name_run = f"MultiSenseX_fewshot_{var_r}_{env_name}_k{var_few_shot_ratio}"
        wandb.init(
            project="multiSenseX",
            name=name_run,
            config=preset,
            reinit=True
        )
        #
        torch.random.manual_seed(var_r + 39)
        #
        ## ---------------------------------------- Teacher ----------------------------------------
        #
        teacher = MultiSenseX(var_x_shape, embedding_dim=100, threshold=0.5).to(device)
        teacher_optimizer = torch.optim.Adam(teacher.parameters(),
                                             lr=preset["nn"]["lr"],
                                             weight_decay=preset["nn"]["weight_decay"])
        teacher_time_0 = time.time()
        teacher_best_weight = train(model=teacher,
                                    optimizer=teacher_optimizer,
                                    data_train_set=teacher_train_set,
                                    data_test_set=teacher_valid_set,
                                    var_threshold=preset["nn"]["threshold"],
                                    var_batch_size=preset["nn"]["batch_size"],
                                    var_epochs=var_teacher_epochs,
                                    device=device,
                                    var_mode="multi_senseX",
                                    var_selection="ppp_then_error")
        teacher_time_1 = time.time()
        teacher.load_state_dict(teacher_best_weight)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        #
        ## ---------------------------------------- Student ----------------------------------------
        #
        student = MultiSenseX(var_x_shape, embedding_dim=100, threshold=0.5).to(device)
        student_optimizer = torch.optim.Adam(student.parameters(),
                                             lr=preset["nn"]["lr"],
                                             weight_decay=preset["nn"]["weight_decay"])
        kd_loss = JointDistillationLoss(temperature=var_kd_temperature)
        student_time_0 = time.time()
        student_best_weight = train(model=student,
                                    optimizer=student_optimizer,
                                    data_train_set=student_train_set,
                                    data_test_set=teacher_valid_set,
                                    var_threshold=preset["nn"]["threshold"],
                                    var_batch_size=preset["nn"]["batch_size"],
                                    var_epochs=var_student_epochs,
                                    device=device,
                                    var_mode="multi_senseX",
                                    teacher=teacher,
                                    kd_loss=kd_loss,
                                    kd_weight=var_kd_weight,
                                    var_selection="ppp_then_error")
        student_time_1 = time.time()
        student.load_state_dict(student_best_weight)

        #
        ## ---------------------------- Test on the other environments ----------------------------
        #
        for test_env_name, (X_env, y_act_env, y_loc_env) in test_sets_by_env.items():
            X_env = X_env.reshape(X_env.shape[0], X_env.shape[1], -1)
            env_loader = DataLoader(JointActLocDataset(X_env, y_act_env, y_loc_env),
                                    batch_size=preset["nn"]["batch_size"], shuffle=False)
            act_preds = []
            loc_preds = []
            with torch.no_grad():
                for xb, _, _ in env_loader:
                    p_act, p_loc, _ = student(xb.to(device))
                    act_preds.append(p_act.cpu())
                    loc_preds.append(p_loc.cpu())
            predict_act = torch.cat(act_preds, dim=0).numpy()
            predict_loc = torch.cat(loc_preds, dim=0).numpy()

            act_metrics, loc_metrics = performance_metrics_joint_multiSensX(
                y_act_env, predict_act, y_loc_env, predict_loc)

            if test_env_name not in env_result:
                env_result[test_env_name] = {
                    "act": {k: [] for k in ("accuracy", "PPP", "precision", "recall", "f1_score", "total_error")},
                    "loc": {k: [] for k in ("accuracy", "PPP", "precision", "recall", "f1_score", "total_error")},
                    "time_teacher": [],
                    "time_student": [],
                }
            for key, metrics in (("act", act_metrics), ("loc", loc_metrics)):
                env_result[test_env_name][key]["accuracy"].append(metrics["accuracy"])
                env_result[test_env_name][key]["PPP"].append(metrics["perfect_prediction_percentage"])
                env_result[test_env_name][key]["precision"].append(metrics["precision"])
                env_result[test_env_name][key]["recall"].append(metrics["recall"])
                env_result[test_env_name][key]["f1_score"].append(metrics["f1_score"])
                env_result[test_env_name][key]["total_error"].append(metrics["total_error"])
            env_result[test_env_name]["time_teacher"].append(teacher_time_1 - teacher_time_0)
            env_result[test_env_name]["time_student"].append(student_time_1 - student_time_0)

            wandb.log({
                f"test_results_per_env/{test_env_name}/ACT_accuracy": act_metrics["accuracy"],
                f"test_results_per_env/{test_env_name}/ACT_ppp": act_metrics["perfect_prediction_percentage"],
                f"test_results_per_env/{test_env_name}/ACT_precision": act_metrics["precision"],
                f"test_results_per_env/{test_env_name}/ACT_recall": act_metrics["recall"],
                f"test_results_per_env/{test_env_name}/ACT_f1_score": act_metrics["f1_score"],
                f"test_results_per_env/{test_env_name}/LOC_accuracy": loc_metrics["accuracy"],
                f"test_results_per_env/{test_env_name}/LOC_ppp": loc_metrics["perfect_prediction_percentage"],
                f"test_results_per_env/{test_env_name}/LOC_precision": loc_metrics["precision"],
                f"test_results_per_env/{test_env_name}/LOC_recall": loc_metrics["recall"],
                f"test_results_per_env/{test_env_name}/LOC_f1_score": loc_metrics["f1_score"],
                f"test_results_per_env/{test_env_name}/teacher_train_time": teacher_time_1 - teacher_time_0,
                f"test_results_per_env/{test_env_name}/student_train_time": student_time_1 - student_time_0,
            }, step=var_r + 100000)

            print(f"  [{test_env_name}] ACT PPP {act_metrics['perfect_prediction_percentage']:.4f} "
                  f"F1 {act_metrics['f1_score']:.4f} | LOC PPP {loc_metrics['perfect_prediction_percentage']:.4f} "
                  f"F1 {loc_metrics['f1_score']:.4f}")

        del teacher_optimizer, student_optimizer, kd_loss
        del teacher_best_weight, student_best_weight, env_loader, act_preds, loc_preds
        torch.cuda.empty_cache()
        gc.collect()
        del student, teacher

    #
    ## -------------------------------------- Aggregate per environment ----------------------------------------
    #
    all_envs_results = {}
    for test_env_name, per_env in env_result.items():
        all_envs_results[test_env_name] = {}
        for key in ("act", "loc"):
            stats = {}
            for metric_name, values in per_env[key].items():
                mean, std, se = _mean_std_se(np.array(values))
                stats[f"avg_{metric_name}"] = mean
                stats[f"std_{metric_name}"] = std
                stats[f"se_{metric_name}"] = se
            stats["avg_teacher_train_time"] = float(np.mean(per_env["time_teacher"]))
            stats["avg_student_train_time"] = float(np.mean(per_env["time_student"]))
            all_envs_results[test_env_name][key] = stats

        act = all_envs_results[test_env_name]["act"]
        loc = all_envs_results[test_env_name]["loc"]
        print(f"\n[{test_env_name}] avg over {var_repeat} repeats:")
        print(f"  ACTIVITY: PPP {act['avg_PPP']:.4f} ± {act['se_PPP']:.4f} | "
              f"Accuracy {act['avg_accuracy']:.4f} ± {act['se_accuracy']:.4f} | "
              f"Total Error {act['avg_total_error']:.4f} ± {act['se_total_error']:.4f} | "
              f"Precision {act['avg_precision']:.4f} | Recall {act['avg_recall']:.4f} | "
              f"F1 {act['avg_f1_score']:.4f}")
        print(f"  LOCATION: PPP {loc['avg_PPP']:.4f} ± {loc['se_PPP']:.4f} | "
              f"Accuracy {loc['avg_accuracy']:.4f} ± {loc['se_accuracy']:.4f} | "
              f"Total Error {loc['avg_total_error']:.4f} ± {loc['se_total_error']:.4f} | "
              f"Precision {loc['avg_precision']:.4f} | Recall {loc['avg_recall']:.4f} | "
              f"F1 {loc['avg_f1_score']:.4f}")

    wandb.finish()
    return all_envs_results


## ====================================================================================================================
# MAIN RUNNER (run_main.py interface)

def _summarize_metrics(var_ppp, var_total_error, var_precision, var_recall, var_f1_score, var_accuracy):
    """
    [description]
    : average/SE summary over the repeats for one task, in the shape format_result() expects
    """
    stats = {}
    for metric_name, values in (("PPP", var_ppp), ("total_error", var_total_error),
                                ("precision", var_precision), ("recall", var_recall),
                                ("f1_score", var_f1_score), ("accuracy", var_accuracy)):
        mean, _, se = _mean_std_se(np.array(values))
        stats[f"avg_{metric_name}"] = mean
        stats[f"se_{metric_name}"] = se
    return stats


def run_multi_senseX(data_train_x,
                     data_train_y,
                     data_test_x,
                     data_test_y,
                     var_repeat=10, var_task="location", var_env="empty_room",
                     save_path="./visualizations/temp"):
    """
    [description]
    : run WiFi-based location model MultiSenseX (location-only).
    [parameter]
    : data_train_x: numpy array, CSI amplitude to train model
    : data_train_y: numpy array, location targets of shape (N, 5) with 0/1 entries
    : data_test_x: numpy array, CSI amplitude to test model
    : data_test_y: numpy array, location targets, same shape as data_train_y
    : var_repeat: int, number of repeated experiments
    : var_task: str, task name kept for interface compatibility (the model is location-only)
    : var_env: str, environment name used for the run name
    : save_path: str, directory kept for symmetry with the other runners (no figures written)
    [return]
    : result: dict, averaged location metrics with SE (single-model shape for format_result)
    """
    #
    ##
    data_train_y = np.asarray(data_train_y, dtype=np.float32)
    data_test_y = np.asarray(data_test_y, dtype=np.float32)

    # Update device selection to check for CUDA first, then MPS (Apple Silicon), then CPU
    device = select_device()
    print(f"Using device: {device}")

    #
    ## ============================================ Preprocess ============================================
    #
    data_valid_x, data_test_x, data_valid_y, data_test_y = train_test_split(
        data_test_x, data_test_y, test_size=0.5, shuffle=True, random_state=39)
    data_valid_x = data_valid_x.reshape(data_valid_x.shape[0], data_valid_x.shape[1], -1)
    data_train_x = data_train_x.reshape(data_train_x.shape[0], data_train_x.shape[1], -1)
    data_test_x = data_test_x.reshape(data_test_x.shape[0], data_test_x.shape[1], -1)
    #
    ## shape for model
    var_x_shape = data_train_x[0].shape
    #
    data_train_set = TensorDataset(torch.from_numpy(data_train_x), torch.from_numpy(data_train_y))
    data_valid_set = TensorDataset(torch.from_numpy(data_valid_x), torch.from_numpy(data_valid_y))
    data_test_set = TensorDataset(torch.from_numpy(data_test_x), torch.from_numpy(data_test_y))
    #
    ##
    ## ========================================= Train & Evaluate =========================================
    #
    result_ppp, result_total_error, result_precision = [], [], []
    result_recall, result_f1_score, result_accuracy = [], [], []
    result_time_train, result_time_test = [], []

    var_macs, var_params = get_model_complexity_info(MultiSenseX(var_x_shape, location_only=True),
                                                     var_x_shape, as_strings=False)
    print("Parameters:", var_params, "- FLOPs:", var_macs * 2)

    for var_r in range(var_repeat):
        #
        ##
        print("Repeat", var_r)
        name_run = f"MultiSenseX{var_r}_" + "_".join(var_env)
        wandb.init(
            project="multiSenseX",
            name=name_run,
            config=preset,
            reinit=True 
        )
        #
        torch.random.manual_seed(var_r + 39)
        #
        model_multiSenseX = MultiSenseX(var_x_shape,
                                        embedding_dim=100,
                                        threshold=0.5,
                                        location_only=True).to(device)
        optimizer = torch.optim.Adam(model_multiSenseX.parameters(),
                                     lr=preset["nn"]["lr"],
                                     weight_decay=preset["nn"]["weight_decay"])
        #
        var_time_0 = time.time()
        #
        ## ---------------------------------------- Train -----------------------------------------
        #
        var_best_weight = train_location(model=model_multiSenseX,
                                         optimizer=optimizer,
                                         data_train_set=data_train_set,
                                         data_test_set=data_valid_set,
                                         var_batch_size=preset["nn"]["batch_size"],
                                         var_epochs=preset["nn"]["epoch"],
                                         device=device)
        var_time_1 = time.time()

        ##
        ## ---------------------------------------- Test ------------------------------------------
        #
        model_multiSenseX.load_state_dict(var_best_weight)
        test_loader = torch.utils.data.DataLoader(data_test_set,
                                                  batch_size=preset["nn"]["batch_size"],
                                                  shuffle=False)
        preds = []
        with torch.no_grad():
            for xb, _ in test_loader:
                preds.append(model_multiSenseX(xb.to(device))[1].cpu())
        predict_test_y = torch.cat(preds, dim=0).numpy()
        #
        var_time_2 = time.time()
        #
        ## -------------------------------------- Evaluate ----------------------------------------
        #
        ##

        layers_idxs = ["layer_"+str(i) for i in range(preset["nn"]["num_decoder_layers"])]
        last_layer_only = True
        # Store results for each layer
        all_layers_results = {}
        dict_layer_acc = performance_metrics(data_test_y, predict_test_y, var_mode="multi_head")

        # Process each layer separately
        for idx, layer_idx in enumerate(layers_idxs):
            layer_metrics = dict_layer_acc[layer_idx]
            if var_r == 0:  # Initialize lists on first repeat
                result_ppp.append([])
                result_time_train.append([])
                result_time_test.append([])
                result_total_error.append([])
                result_precision.append([])
                result_recall.append([])
                result_f1_score.append([])
                result_avg_count_error.append([])
                result_accuracy.append([])
            result_accuracy[idx].append(layer_metrics['accuracy'])
            result_ppp[idx].append(layer_metrics['perfect_prediction_percentage'])
            result_time_train[idx].append(var_time_1 - var_time_0)
            result_time_test[idx].append(var_time_2 - var_time_1)
            result_total_error[idx].append(layer_metrics['total_error'])
            result_precision[idx].append(layer_metrics['precision'])
            result_recall[idx].append(layer_metrics['recall'])
            result_f1_score[idx].append(layer_metrics['f1_score'])
            result_avg_count_error[idx].append(layer_metrics['mean_count_error'])

        if last_layer_only:
            layer_metrics = dict_layer_acc["layer_" +str(preset["nn"]["num_decoder_layers"] - 1)]
            wandb.log({
                f"test_results/repeat": var_r,
                f"test_results/train_time": var_time_1 - var_time_0,
                f"test_results/test_time": var_time_2 - var_time_1,
                f"test_results/TOTAL_TESTSET_ERROR": layer_metrics['total_error'],
                f"test_results/TOTAL_TESTSET_perfect_prediction_percentage": layer_metrics[
                    'perfect_prediction_percentage'],
                f"test_results/TOTAL_ACCURACY": layer_metrics['accuracy'],
                f"test_results/mean_count_error": layer_metrics['mean_count_error'],
                f"test_results/error_per_person_1": layer_metrics['error_per_person'][0],
                f"test_results/error_per_person_2": layer_metrics['error_per_person'][1],
                f"test_results/error_per_person_3": layer_metrics['error_per_person'][2],
                f"test_results/error_per_person_4": layer_metrics['error_per_person'][3],
                f"test_results/error_per_person_5": layer_metrics['error_per_person'][4],
                f"test_results/precision": layer_metrics['precision'],
                f"test_results/recall": layer_metrics['recall'],
                f"test_results/f1_score": layer_metrics['f1_score']
            }, step=var_r + 100000)

            print(
                "- Total Error %.6f" % layer_metrics['total_error'],
                "- Perfect Prediction Percentage %.6f" % layer_metrics['perfect_prediction_percentage'])
            
        del optimizer, loss
        torch.cuda.empty_cache()
        gc.collect()
        if var_r == var_repeat - 1:
            last_model = model_AMAR_WO_RVQ   
        else:
            del model_AMAR_WO_RVQ
    # Calculate averages and standard errors for each layer
    for layer_idx_num, layer_idx in enumerate(layers_idxs):
        # Calculate metrics with standard errors
        ppp_array = np.array(result_ppp[layer_idx_num])
        precision_array = np.array(result_precision[layer_idx_num])
        recall_array = np.array(result_recall[layer_idx_num])
        f1_array = np.array(result_f1_score[layer_idx_num])
        accuracy_array = np.array(result_accuracy[layer_idx_num])
        total_error_array = np.array(result_total_error[layer_idx_num])
        
        # Store results for this layer
        all_layers_results[layer_idx] = {
            'avg_PPP': float(np.mean(ppp_array)),
            'avg_precision': float(np.mean(precision_array)),
            'avg_recall': float(np.mean(recall_array)),
            'avg_f1_score': float(np.mean(f1_array)),
            'avg_accuracy': float(np.mean(accuracy_array)),
            'avg_total_error': float(np.mean(total_error_array)),
            'std_PPP': float(np.std(ppp_array, ddof=1)) if len(ppp_array) > 1 else 0.0,
            'std_precision': float(np.std(precision_array, ddof=1)) if len(precision_array) > 1 else 0.0,
            'std_recall': float(np.std(recall_array, ddof=1)) if len(recall_array) > 1 else 0.0,
            'std_f1_score': float(np.std(f1_array, ddof=1)) if len(f1_array) > 1 else 0.0,
            'std_accuracy': float(np.std(accuracy_array, ddof=1)) if len(accuracy_array) > 1 else 0.0,
            'std_total_error': float(np.std(total_error_array, ddof=1)) if len(total_error_array) > 1 else 0.0,
            'se_PPP': float(np.std(ppp_array, ddof=1) / np.sqrt(len(ppp_array))) if len(ppp_array) > 1 else 0.0,
            'se_precision': float(np.std(precision_array, ddof=1) / np.sqrt(len(precision_array))) if len(precision_array) > 1 else 0.0,
            'se_recall': float(np.std(recall_array, ddof=1) / np.sqrt(len(recall_array))) if len(recall_array) > 1 else 0.0,
            'se_f1_score': float(np.std(f1_array, ddof=1) / np.sqrt(len(f1_array))) if len(f1_array) > 1 else 0.0,
            'se_accuracy': float(np.std(accuracy_array, ddof=1) / np.sqrt(len(accuracy_array))) if len(accuracy_array) > 1 else 0.0,
            'se_total_error': float(np.std(total_error_array, ddof=1) / np.sqrt(len(total_error_array))) if len(total_error_array) > 1 else 0.0
        }
        
        wandb.log({
            f"test_results/{layer_idx}/avg_PPP": all_layers_results[layer_idx]['avg_PPP'],
            f"test_results/{layer_idx}/avg_train_time": sum(result_time_train[layer_idx_num]) / len(result_time_train[layer_idx_num]),
            f"test_results/{layer_idx}/avg_test_time": sum(result_time_test[layer_idx_num]) / len(result_time_test[layer_idx_num]),
            f"test_results/{layer_idx}/avg_total_error": all_layers_results[layer_idx]['avg_total_error'],
            f"test_results/{layer_idx}/avg_precision": all_layers_results[layer_idx]['avg_precision'],
            f"test_results/{layer_idx}/avg_recall": all_layers_results[layer_idx]['avg_recall'],
            f"test_results/{layer_idx}/avg_f1_score": all_layers_results[layer_idx]['avg_f1_score'],
            f"test_results/{layer_idx}/avg_count_error": sum(result_avg_count_error[layer_idx_num]) / len(result_avg_count_error[layer_idx_num]),
            f"test_results/{layer_idx}/avg_accuracy": all_layers_results[layer_idx]['avg_accuracy']
        })  # Use an even larger offset for averages
    
    # Use the last layer for visualization and final results
    last_layer = layers_idxs[-1]
    last_layer_predictions = predict_test_y[last_layer] if isinstance(predict_test_y, dict) else predict_test_y
    # dict_true_acc = all_layers_results[last_layer]

    # Run visualization with the last layer's predictions
    
    log_random_attention_weights_final(last_model, np.argmax(predict_test_y[-1], axis=-1), np.argmax(data_test_y, axis=-1), 1000000000, 50, var_task)
    
    viz_stats = visualize_model_performance(
        y_pred=last_layer_predictions,
        y_true=data_test_y,
        var_mode=var_mode,
        save_dir=save_path
    )

    wandb.finish()
    return all_layers_results
