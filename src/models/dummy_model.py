"""
[file]          dummy_model.py
"""
#
##
import copy
import os
import time
import torch
import numpy as np
#
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from ptflops import get_model_complexity_info
from sklearn.metrics import classification_report, accuracy_score
#
from src.train import train
from configs.preset import preset
from src.utils import *
import wandb


class DUMMY(nn.Module):
    """
    [description]
    : The "most average output" baseline. It sees the CSI -- forward takes the flattened window, the
      same interface every real model uses -- but it is trained with a mean-squared-error objective,
      whose least-squares optimum is the per-location mean of the training outputs. With no spatial
      prior to fit, the model converges to that mean, so it predicts the same "most average" output
      vector for every frame and every test room: the constant reference a real model must beat.
    """

    def __init__(self, var_x_shape, var_y_shape):
        super(DUMMY, self).__init__()

        # The input is flattened to one vector per sample (see forward), so every non-batch axis
        # contributes: (N, 3000, 3, 3, 30) -> 3000*3*3*30 = 810000 features. Using var_x_shape[1:]
        # dropped one axis (270 = 3*3*30) and never matched the flattened input.
        self.num_features = int(np.prod(var_x_shape))
        self.num_classes = int(np.prod(var_y_shape))

        # A single linear map. Trained with MSE it learns the conditional mean occupancy per
        # location; for a baseline with no useful inductive bias that is the training-set mean, i.e.
        # the one output that best covers every test frame.
        self.layer_output = nn.Linear(self.num_features, self.num_classes)

    def forward(self, var_input):
        """
        [return]
        : output: (batch, num_classes) predicted per-location occupancy; the "most average" output.
        """
        var_x = var_input.reshape(var_input.size(0), -1)
        return self.layer_output(var_x)
#
##
def run_DUMMY(data_train_x,
              data_train_y,
              data_test_x,
              data_test_y,
              var_repeat=10, var_task="location", var_env="empty_room", save_path="./visualizations/temp"):
    #
    device = select_device()

    data_train_x = np.asarray(data_train_x)
    data_train_y = np.asarray(data_train_y)
    data_test_x = np.asarray(data_test_x)
    data_test_y = np.asarray(data_test_y)

    if data_train_x.ndim == 2:
        data_train_x = data_train_x[:, None, :]
    if data_test_x.ndim == 2:
        data_test_x = data_test_x[:, None, :]

    if data_train_y.ndim == 1:
        data_train_y = data_train_y[:, None]
    if data_test_y.ndim == 1:
        data_test_y = data_test_y[:, None]

    var_x_shape = data_train_x.shape[1:]

    var_y_shape = ([data_train_y.sum(axis=1).shape[1]] if data_train_y.ndim == 3
                   else [data_train_y.shape[1]])

    ## The dummy trains on the training split and is scored on the held-out test split, like every
    ## other model. The metric sums the 6-user one-hot label to a per-location count vector, so the
    ## dummy regresses that same vector; MSE then drives it to the mean per-location count.
    var_train_target = data_train_y.sum(axis=1) if data_train_y.ndim == 3 else data_train_y
    var_valid_x, var_train_x, var_valid_y, var_train_y = train_test_split(
        data_train_x, var_train_target, test_size=0.1, shuffle=True, random_state=39)
    var_train_loader = DataLoader(
        TensorDataset(torch.from_numpy(var_train_x), torch.from_numpy(var_train_y)),
        batch_size=preset["nn"]["batch_size"], shuffle=True)
    var_valid_loader = DataLoader(
        TensorDataset(torch.from_numpy(var_valid_x), torch.from_numpy(var_valid_y)),
        batch_size=preset["nn"]["batch_size"], shuffle=False)
    ## The empirical mean output, printed next to the trained model's prediction as a sanity check:
    ## a converged dummy reproduces this vector whatever the input.
    var_mean_output = var_train_target.mean(axis=0)

    result = {}
    result_accuracy = []
    result_time_train = []
    result_time_test = []
    result_total_error = []
    result_precision = []
    result_recall = []
    result_f1_score = []

    for var_r in range(var_repeat):
        print("Repeat", var_r)

        var_time_0 = time.time()
        model_dummy = DUMMY(var_x_shape, var_y_shape).to(device)
        optimizer = torch.optim.Adam(model_dummy.parameters(), lr=preset["nn"]["lr"],
                                     weight_decay=preset["nn"]["weight_decay"])
        ## Model selection on the validation split, exactly as the real runners do, so the reported
        ## test metrics come from the epoch that best reproduces the mean output.
        var_best_loss, var_best_weight = float("inf"), None
        for var_epoch in range(preset["nn"]["epoch"]):
            model_dummy.train()
            for var_xb, var_yb in var_train_loader:
                optimizer.zero_grad()
                var_loss = F.mse_loss(model_dummy(var_xb.float().to(device)),
                                      var_yb.float().to(device))
                var_loss.backward()
                optimizer.step()
            model_dummy.eval()
            with torch.no_grad():
                var_valid_loss = float(np.mean([
                    float(F.mse_loss(model_dummy(var_xb.float().to(device)),
                                     var_yb.float().to(device)))
                    for var_xb, var_yb in var_valid_loader]))
            if var_valid_loss < var_best_loss:
                var_best_loss = var_valid_loss
                var_best_weight = copy.deepcopy(model_dummy.state_dict())
        model_dummy.load_state_dict(var_best_weight)
        var_time_1 = time.time()

        model_dummy.eval()
        with torch.no_grad():
            x = torch.from_numpy(data_test_x).float().to(device)
            predict_test_y = model_dummy(x).detach().cpu().numpy()
        if var_r == 0:
            print(f"  mean training output : {np.round(var_mean_output, 4).tolist()}")
            print(f"  dummy mean prediction: {np.round(predict_test_y.mean(axis=0), 4).tolist()}")

        var_time_2 = time.time()

        data_test_y_c = data_test_y.sum(axis=1)
        dict_true_acc = performance_metrics(data_test_y_c, predict_test_y, var_mode="count_classification")

        if wandb.run is not None:
            wandb.log({
                "repeat": var_r,
                "train_time": var_time_1 - var_time_0,
                "test_time": var_time_2 - var_time_1,
                "TOTAL_TESTSET_ERROR": dict_true_acc['total_error'],
                "TOTAL_TESTSET_perfect_prediction_percentage": dict_true_acc['perfect_prediction_percentage'],
                "TOTAL_ACCURACY": dict_true_acc['accuracy'],
                "mean_count_error": dict_true_acc['mean_count_error'],
                "precision": dict_true_acc['precision'],
                "recall": dict_true_acc['recall'],
                "f1_score": dict_true_acc['f1_score'],
            })

        result_accuracy.append(dict_true_acc['perfect_prediction_percentage'])
        result_time_train.append(var_time_1 - var_time_0)
        result_time_test.append(var_time_2 - var_time_1)
        result_total_error.append(dict_true_acc['total_error'])
        result_precision.append(dict_true_acc['precision'])
        result_recall.append(dict_true_acc['recall'])
        result_f1_score.append(dict_true_acc['f1_score'])

    ppp_array = np.array(result_accuracy)
    precision_array = np.array(result_precision)
    recall_array = np.array(result_recall)
    f1_array = np.array(result_f1_score)
    total_error_array = np.array(result_total_error)

    aggregated_result = {
        'avg_PPP': float(np.mean(ppp_array)),
        'avg_precision': float(np.mean(precision_array)),
        'avg_recall': float(np.mean(recall_array)),
        'avg_f1_score': float(np.mean(f1_array)),
        'avg_total_error': float(np.mean(total_error_array)),
        'std_PPP': float(np.std(ppp_array, ddof=1)) if len(ppp_array) > 1 else 0.0,
        'std_precision': float(np.std(precision_array, ddof=1)) if len(precision_array) > 1 else 0.0,
        'std_recall': float(np.std(recall_array, ddof=1)) if len(recall_array) > 1 else 0.0,
        'std_f1_score': float(np.std(f1_array, ddof=1)) if len(f1_array) > 1 else 0.0,
        'std_total_error': float(np.std(total_error_array, ddof=1)) if len(total_error_array) > 1 else 0.0,
        'se_PPP': float(np.std(ppp_array, ddof=1) / np.sqrt(len(ppp_array))) if len(ppp_array) > 1 else 0.0,
        'se_precision': float(np.std(precision_array, ddof=1) / np.sqrt(len(precision_array))) if len(precision_array) > 1 else 0.0,
        'se_recall': float(np.std(recall_array, ddof=1) / np.sqrt(len(recall_array))) if len(recall_array) > 1 else 0.0,
        'se_f1_score': float(np.std(f1_array, ddof=1) / np.sqrt(len(f1_array))) if len(f1_array) > 1 else 0.0,
        'se_total_error': float(np.std(total_error_array, ddof=1) / np.sqrt(len(total_error_array))) if len(total_error_array) > 1 else 0.0,
        'avg_train_time': float(np.mean(result_time_train)),
        'avg_test_time': float(np.mean(result_time_test)),
    }


    ## The head emits a per-location count vector (N, num_locations), exactly the target the
    ## count_classification metric scored above, so the figures must use the same mode and the same
    ## summed target. multi_head expects 3-D per-layer one-hot predictions (y_pred[-1] unpacked as
    ## batch x heads x classes) and crashed on the 2-D count vector.
    visualize_model_performance(
        y_pred=predict_test_y,
        y_true=data_test_y_c,
        var_mode="count_classification",
        save_dir=save_path
    )

    if wandb.run is not None:
        wandb.log({
            "avg_accuracy": aggregated_result['avg_PPP'],
            "avg_train_time": aggregated_result['avg_train_time'],
            "avg_test_time": aggregated_result['avg_test_time'],
            "avg_total_error": aggregated_result['avg_total_error'],
            "avg_precision": aggregated_result['avg_precision'],
            "avg_recall": aggregated_result['avg_recall'],
            "avg_f1_score": aggregated_result['avg_f1_score'],
        })

    return aggregated_result
#
##
def run_dummy_cross_domain(train_sets_by_env,
                           test_sets_by_env,
                           var_repeat=10, var_task="location", var_env="empty_room",
                           save_path="./visualizations/temp"):
    """
    [description]
    : leave-one-room-out run of the "most average output" dummy. The dummy is trained on the training
      rooms' frames against their summed per-location occupancy and evaluated on every held-out room.
      Its MSE optimum is the mean training output, so it predicts the same "most average" vector for
      every test room -- the constant reference a real cross-domain model must beat. A model whose
      held-out metrics match this one has not learned a transferable CSI-to-location mapping; it is
      emitting the training rooms' location prior.
    : train_sets_by_env / test_sets_by_env: dict env -> (X, y). y is the raw label; it is summed over
      its user axis to the per-location occupancy the metric scores.
    : return: dict, env -> averaged metrics with SE, plus the dummy's mean output, the mean training
      output and the test room's own mean target (all in the JSON), so "is the model just
      reproducing the training prior?" can be answered without re-running.
    """
    #
    device = select_device()
    print(f"Using device: {device}")
    #
    ## ------------------------------------------ data prep -----------------------------------------
    ## Each training room keeps its CSI in place (torch.from_numpy shares the buffer) and the 90/10
    ## split is a Subset over indices, so a multi-room run costs no extra CSI memory -- the same
    ## pattern run_density_map_cross_domain and the AMAR cross-domain runner use.
    var_train_datasets, var_valid_datasets = [], []
    var_x_shape, var_num_classes = None, None
    var_train_targets = []
    for var_env_name, (var_x_env, var_y_env) in train_sets_by_env.items():
        var_x_env = np.asarray(var_x_env).reshape(var_x_env.shape[0], var_x_env.shape[1], -1)
        var_y_env = np.asarray(var_y_env)
        var_y_env = var_y_env.sum(axis=1) if var_y_env.ndim == 3 else var_y_env
        if var_x_shape is None:
            var_x_shape, var_num_classes = var_x_env.shape[1:], var_y_env.shape[1]
        var_train_targets.append(var_y_env)
        var_dataset = TensorDataset(torch.from_numpy(var_x_env), torch.from_numpy(var_y_env))
        var_perm = np.random.RandomState(39).permutation(len(var_dataset))
        var_num_valid = max(1, int(round(0.1 * len(var_dataset))))
        var_valid_idx = np.sort(var_perm[:var_num_valid])
        var_train_idx = np.sort(var_perm[var_num_valid:])
        var_train_datasets.append(torch.utils.data.Subset(var_dataset, var_train_idx.tolist()))
        var_valid_datasets.append(torch.utils.data.Subset(var_dataset, var_valid_idx.tolist()))
    data_train_set = torch.utils.data.ConcatDataset(var_train_datasets)
    data_valid_set = torch.utils.data.ConcatDataset(var_valid_datasets)
    ## the mean training output: what a converged dummy predicts for every test room
    var_train_mean = np.concatenate(var_train_targets).mean(axis=0)
    print(f"Training rooms: {list(train_sets_by_env)} ({len(data_train_set)} train / "
          f"{len(data_valid_set)} valid) | Test rooms: {list(test_sets_by_env)}")
    print(f"Mean training output: {np.round(var_train_mean, 4).tolist()}")
    #
    env_rep_metrics, env_last_pred = {}, {}
    #
    for var_r in range(var_repeat):
        print("Repeat", var_r)
        torch.random.manual_seed(var_r + 39)
        var_train_loader = DataLoader(data_train_set, batch_size=preset["nn"]["batch_size"], shuffle=True)
        var_valid_loader = DataLoader(data_valid_set, batch_size=preset["nn"]["batch_size"], shuffle=False)
        #
        model_dummy = DUMMY(var_x_shape, [var_num_classes]).to(device)
        optimizer = torch.optim.Adam(model_dummy.parameters(), lr=preset["nn"]["lr"],
                                     weight_decay=preset["nn"]["weight_decay"])
        var_best_loss, var_best_weight = float("inf"), None
        for var_epoch in range(preset["nn"]["epoch"]):
            model_dummy.train()
            for var_xb, var_yb in var_train_loader:
                optimizer.zero_grad()
                var_loss = F.mse_loss(model_dummy(var_xb.float().to(device)),
                                      var_yb.float().to(device))
                var_loss.backward()
                optimizer.step()
            model_dummy.eval()
            with torch.no_grad():
                var_valid_loss = float(np.mean([
                    float(F.mse_loss(model_dummy(var_xb.float().to(device)),
                                     var_yb.float().to(device)))
                    for var_xb, var_yb in var_valid_loader]))
            if var_valid_loss < var_best_loss:
                var_best_loss = var_valid_loss
                var_best_weight = copy.deepcopy(model_dummy.state_dict())
        model_dummy.load_state_dict(var_best_weight)
        model_dummy.eval()
        #
        ## --------------------------------------- test per room ------------------------------------
        for var_env_name, (var_x_env, var_y_env) in test_sets_by_env.items():
            var_x_env = np.asarray(var_x_env).reshape(var_x_env.shape[0], var_x_env.shape[1], -1)
            var_y_env = np.asarray(var_y_env)
            var_y_env = var_y_env.sum(axis=1) if var_y_env.ndim == 3 else var_y_env
            var_loader = DataLoader(TensorDataset(torch.from_numpy(var_x_env)),
                                    batch_size=preset["nn"]["batch_size"], shuffle=False)
            var_pred = []
            with torch.no_grad():
                for (var_xb,) in var_loader:
                    var_pred.append(model_dummy(var_xb.float().to(device)).cpu())
            var_pred = torch.cat(var_pred, dim=0).numpy()
            var_metrics = performance_metrics(var_y_env, var_pred, var_mode="count_classification")
            env_rep_metrics.setdefault(var_env_name, []).append(var_metrics)
            env_last_pred[var_env_name] = (var_pred, var_y_env)
            print(f"  [{var_env_name}] Total Error {var_metrics['total_error']:.4f} - "
                  f"PPP {var_metrics['perfect_prediction_percentage']:.2f}% - "
                  f"Acc {var_metrics['accuracy']:.4f} - F1 {var_metrics['f1_score']:.4f}")
    #
    ## -------------------------------------- aggregate per room ------------------------------------
    results = {}
    for var_env_name, var_rep_list in env_rep_metrics.items():
        var_env_result = {}
        for var_key in ("accuracy", "precision", "recall", "f1_score",
                        "perfect_prediction_percentage", "total_error", "mean_count_error"):
            var_arr = np.array([var_rep[var_key] for var_rep in var_rep_list], dtype=float)
            var_env_result[f"avg_{var_key}"] = float(var_arr.mean())
            var_std = float(var_arr.std(ddof=1)) if len(var_arr) > 1 else 0.0
            var_env_result[f"std_{var_key}"] = var_std
            var_env_result[f"se_{var_key}"] = var_std / np.sqrt(len(var_arr)) if len(var_arr) > 1 else 0.0
        ## the generic format_result of the cross-domain script reads avg_PPP / avg_total_error /
        ## avg_accuracy / ..., so alias the perfect-prediction rate under that name
        var_env_result["avg_PPP"] = var_env_result["avg_perfect_prediction_percentage"]
        var_env_result["se_PPP"] = var_env_result["se_perfect_prediction_percentage"]
        var_env_result["std_PPP"] = var_env_result["std_perfect_prediction_percentage"]
        ## the constant the dummy emits, the mean training output and the test room's own mean
        ## target: if dummy_mean_output ~ train_mean_output and both differ from test_mean_target,
        ## the dummy (and any model that matches it) is reproducing the training rooms' prior.
        var_pred_last, var_true_last = env_last_pred[var_env_name]
        var_env_result["dummy_mean_output"] = np.round(var_pred_last.mean(axis=0), 6).tolist()
        var_env_result["train_mean_output"] = np.round(var_train_mean, 6).tolist()
        var_env_result["test_mean_target"] = np.round(var_true_last.mean(axis=0), 6).tolist()
        results[var_env_name] = var_env_result
        ## the standard per-location figures on the last repeat
        visualize_model_performance(y_pred=var_pred_last, y_true=var_true_last,
                                    var_mode="count_classification",
                                    save_dir=os.path.join(str(save_path), var_env_name))
    #
    if wandb.run is not None:
        wandb.log({f"dummy/{var_env}/{var_key}": var_value
                   for var_env, var_res in results.items()
                   for var_key, var_value in var_res.items() if isinstance(var_value, float)})
    #
    return results
