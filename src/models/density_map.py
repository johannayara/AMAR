"""
[file]          density_map.py
[description]   Room-agnostic group counting from WiFi CSI via a spatial density map.

                The model predicts a density map over the normalized room frame defined by
                preset["layouts"] (from the WiMANS environment layouts). The integral of the map is
                the number of people, and its peaks are the (approximate) locations of the people.
                Because the frame is shared across rooms, "where" means the same physical spot in
                every room, which is what makes counting transferable.
"""
#
##

import copy
import gc
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import center_of_mass, label, maximum_filter
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import TensorDataset
from ptflops import get_model_complexity_info

from src.models.bce_that import THAT
from configs.preset import preset
from src.utils import *
import wandb


torch.set_float32_matmul_precision("high")

#
##
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


class DensityMapNet(torch.nn.Module):
    """
    [description]
    : CSI backbone + two heads: a scalar count head and a spatial head that produces a distribution
      over the grid. The density map is the distribution scaled by the count, so its integral is
      exactly the predicted number of people and its peaks are the predicted locations.
    """
    #
    ##
    def __init__(self,
                 var_x_shape,
                 embedding_dim=100,
                 grid_size=32,
                 hidden_dim=256):

        super().__init__()
        self.backbone = THAT(var_x_shape, [embedding_dim])
        self.grid_size = grid_size

        self.count_head = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.spatial_head = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, grid_size * grid_size),
        )

    def forward(self, x):
        z = self.backbone(x)
        #
        ## count: (batch,) non-negative
        var_count = F.softplus(self.count_head(z)).squeeze(-1)
        #
        ## spatial distribution: (batch, grid, grid), sums to 1
        var_spatial = F.softmax(self.spatial_head(z), dim=-1).view(-1, self.grid_size, self.grid_size)
        #
        ## density: integrates to the predicted count
        var_density = var_spatial * var_count.view(-1, 1, 1)
        #
        return var_density, var_count


#
## ---------------------------------------------------------------------------------------------- ##
## --------------------------------------------- metrics ---------------------------------------- ##
#
##
def _extract_peaks(var_map, var_threshold_frac):
    """
    [description]
    : return the (x, y) normalized coordinates of the local maxima of a single density map.
    """
    #
    var_max = float(var_map.max())
    if var_max <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    #
    var_mask = (var_map == maximum_filter(var_map, size=3)) & (var_map > var_threshold_frac * var_max)
    var_labels, var_num = label(var_mask)
    if var_num == 0:
        return np.zeros((0, 2), dtype=np.float32)
    #
    var_height, var_width = var_map.shape
    var_centroids = center_of_mass(var_map, var_labels, range(1, var_num + 1))
    #
    return np.array([[var_c[1] / var_width, var_c[0] / var_height] for var_c in var_centroids], dtype=np.float32)


def count_metrics(var_true_count, var_pred_count, var_num_classes=6):
    """
    [description]
    : group-count metrics: exact-count accuracy, count MAE, occupancy accuracy/F1 and per-count
      accuracy.
    """
    #
    var_true_count = np.asarray(var_true_count).astype(int)
    var_pred_count = np.asarray(var_pred_count).astype(int)
    #
    var_exact = float(np.mean(var_true_count == var_pred_count))
    var_mae = float(np.mean(np.abs(var_true_count - var_pred_count)))
    #
    var_true_occ = var_true_count > 0
    var_pred_occ = var_pred_count > 0
    var_occ_acc = float(np.mean(var_true_occ == var_pred_occ))
    var_tp = float(np.sum(var_true_occ & var_pred_occ))
    var_fp = float(np.sum(~var_true_occ & var_pred_occ))
    var_fn = float(np.sum(var_true_occ & ~var_pred_occ))
    var_occ_precision = var_tp / (var_tp + var_fp) if (var_tp + var_fp) > 0 else 0.0
    var_occ_recall = var_tp / (var_tp + var_fn) if (var_tp + var_fn) > 0 else 0.0
    var_occ_f1 = (2 * var_occ_precision * var_occ_recall / (var_occ_precision + var_occ_recall)
                  if (var_occ_precision + var_occ_recall) > 0 else 0.0)
    #
    var_per_class = {}
    for var_class in range(var_num_classes):
        var_sel = var_true_count == var_class
        var_per_class[var_class] = float(np.mean(var_pred_count[var_sel] == var_class)) if var_sel.any() else 0.0
    #
    return {
        "accuracy": var_exact,
        "mae": var_mae,
        "occupancy_accuracy": var_occ_acc,
        "occupancy_f1": var_occ_f1,
        "per_class_accuracy": var_per_class,
    }


def localization_metrics(var_true_density, var_pred_density, var_threshold_frac=0.25, var_hit_radius=0.1):
    """
    [description]
    : localization quality of the density map: distance from every true location (peak of the
      ground-truth map) to the nearest predicted peak, and the fraction of true locations that have
      a predicted peak within var_hit_radius (in normalized room units).
    """
    #
    var_distances = []
    var_hits = 0
    var_total = 0
    #
    for var_idx in range(len(var_true_density)):
        var_true_peaks = _extract_peaks(var_true_density[var_idx], var_threshold_frac)
        var_pred_peaks = _extract_peaks(var_pred_density[var_idx], var_threshold_frac)
        #
        for var_true_peak in var_true_peaks:
            var_total += 1
            if len(var_pred_peaks) == 0:
                var_distances.append(1.0)
                continue
            var_dist = np.sqrt(((var_pred_peaks - var_true_peak) ** 2).sum(axis=1)).min()
            var_distances.append(float(var_dist))
            if var_dist <= var_hit_radius:
                var_hits += 1
    #
    return {
        "loc_error": float(np.mean(var_distances)) if var_distances else 0.0,
        "loc_detection": float(var_hits / var_total) if var_total else 0.0,
    }


#
## ---------------------------------------------------------------------------------------------- ##
## --------------------------------------------- training --------------------------------------- ##
#
##
def train_density(model,
                  optimizer,
                  data_train_set: TensorDataset,
                  data_valid_set: TensorDataset,
                  var_batch_size: int,
                  var_epochs: int,
                  device,
                  var_count_loss_weight: float = 1.0,
                  patience: int = 150):
    """
    [description]
    : train the density-map model. Loss = MSE between predicted and target density maps plus an L1
      term on the count (the integral). Model selection is by count MAE, then exact-count accuracy.
    """
    #
    var_train_loader = torch.utils.data.DataLoader(data_train_set, var_batch_size, shuffle=True, pin_memory=True)
    var_valid_loader = torch.utils.data.DataLoader(data_valid_set, len(data_valid_set))
    #
    var_best_mae = np.inf
    var_best_accuracy = 0.0
    var_best_weight = None
    var_epoch_saved = 0
    var_counter = 0
    #
    var_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=preset["nn"]["scheduler"]["num_warmup_epochs"] * len(var_train_loader),
        num_training_steps=preset["nn"]["epoch"] * len(var_train_loader),
        min_lr_ratio=preset["nn"]["scheduler"]["min_lr_ratio"],
    )

    def apply_augmentation(var_x):
        var_x = var_x + torch.randn_like(var_x) * 0.1
        var_scale = torch.rand(var_x.size(0), 1, device=var_x.device) * 0.2 + 0.9
        var_x = var_x * var_scale.unsqueeze(-1)
        var_x = var_x * torch.bernoulli(torch.ones_like(var_x) * 0.96)
        return var_x

    for var_epoch in range(var_epochs):
        var_time_e0 = time.time()
        model.train()
        for var_x, var_y in var_train_loader:
            var_x = apply_augmentation(var_x.to(device))
            var_y = var_y.to(device)
            #
            var_density, var_count = model(var_x)
            var_loss = (F.mse_loss(var_density, var_y)
                        + var_count_loss_weight * F.l1_loss(var_count, var_y.sum(dim=(1, 2))))
            #
            optimizer.zero_grad()
            var_loss.backward()
            optimizer.step()
            var_scheduler.step()
        #
        model.eval()
        with torch.no_grad():
            var_valid_x, var_valid_y = next(iter(var_valid_loader))
            var_pred_density, _ = model(var_valid_x.to(device))
            var_pred_density = var_pred_density.cpu().numpy()
            var_valid_y = var_valid_y.numpy()
        #
        var_metrics = count_metrics(var_valid_y.sum(axis=(1, 2)).round(), var_pred_density.sum(axis=(1, 2)).round())
        #
        wandb.log({
            "epoch": var_epoch,
            "train_loss": var_loss.item(),
            "valid_accuracy": var_metrics["accuracy"],
            "valid_mae": var_metrics["mae"],
            "valid_occupancy_accuracy": var_metrics["occupancy_accuracy"],
            "valid_occupancy_f1": var_metrics["occupancy_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        #
        print(f"Epoch {var_epoch}/{var_epochs} - %.3fs" % (time.time() - var_time_e0),
              "- Loss %.6f" % var_loss.cpu(),
              "- Valid Acc %.4f" % var_metrics["accuracy"],
              "- Valid MAE %.4f" % var_metrics["mae"])
        #
        if (var_metrics["mae"] < var_best_mae
                or (var_metrics["mae"] == var_best_mae and var_metrics["accuracy"] > var_best_accuracy)):
            var_best_mae = var_metrics["mae"]
            var_best_accuracy = var_metrics["accuracy"]
            var_best_weight = copy.deepcopy(model.state_dict())
            var_epoch_saved = var_epoch
            var_counter = 0
        else:
            var_counter += 1
        #
        if var_counter >= patience:
            print(f"Early stopping triggered at epoch {var_epoch}")
            break
    #
    if var_best_weight is None:
        var_best_weight = copy.deepcopy(model.state_dict())
    #
    print(f"Epoch that the model was saved {var_epoch_saved}")
    print(f"Best count MAE: {var_best_mae:.6f}, exact-count accuracy: {var_best_accuracy:.6f}")
    #
    return var_best_weight


#
## ---------------------------------------------------------------------------------------------- ##
## ---------------------------------------- main runner ----------------------------------------- ##
#
##
def run_density_map(data_train_x,
                    data_train_y,
                    data_test_x,
                    data_test_y,
                    var_repeat=10, var_task="count", var_env="empty_room",
                    save_path="./visualizations/temp"):
    """
    [description]
    : run the density-map group-counting model. This is dispatched by scripts/run_main.py, so it
      follows the same call convention as run_AMAR_WO_RVQ:
          run_model(data_train_x, data_train_y, data_test_x, data_test_y,
                    var_repeat, var_task, var_env, save_path)
      Counting and approximate localization are both read off the predicted density map.
    [parameter]
    : data_train_x: numpy array, CSI amplitude to train model
    : data_train_y: numpy array, density targets of shape (N, grid_size, grid_size)
    : data_test_x: numpy array, CSI amplitude to test model
    : data_test_y: numpy array, density targets, same shape as data_train_y
    : var_repeat: int, number of repeated experiments
    : var_task: str, task name kept for interface compatibility (the model is always counting)
    : var_env: str, environment name used for the run name
    : save_path: str, directory kept for symmetry with the other runners (no figures written)
    [return]
    : result: dict, averaged count and localization metrics with SE (group-count shape for
      format_result)
    """
    #
    ##
    data_train_y = np.asarray(data_train_y, dtype=np.float32)
    data_test_y = np.asarray(data_test_y, dtype=np.float32)
    #
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
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
    var_x_shape = data_train_x[0].shape
    var_grid_size = data_train_y.shape[-1]
    #
    data_train_set = TensorDataset(torch.from_numpy(data_train_x), torch.from_numpy(data_train_y))
    data_valid_set = TensorDataset(torch.from_numpy(data_valid_x), torch.from_numpy(data_valid_y))
    data_test_set = TensorDataset(torch.from_numpy(data_test_x), torch.from_numpy(data_test_y))

    #
    ##
    ## ========================================= Train & Evaluate =========================================
    #
    result_accuracy, result_mae, result_occ_accuracy, result_occ_f1 = [], [], [], []
    result_loc_error, result_loc_detection = [], []
    result_per_class = []
    #
    var_macs, var_params = get_model_complexity_info(DensityMapNet(var_x_shape, grid_size=var_grid_size),
                                                     var_x_shape, as_strings=False)
    print("Parameters:", var_params, "- FLOPs:", var_macs * 2)

    for var_r in range(var_repeat):
        #
        ##
        print("Repeat", var_r)
        name_run = f"DensityMap{var_r}_" + "_".join(preset["data"]["environment"])
        wandb.init(project="density_map", name=name_run, config=preset, reinit=True)
        #
        torch.random.manual_seed(var_r + 39)
        #
        model_density = DensityMapNet(var_x_shape, embedding_dim=100, grid_size=var_grid_size).to(device)
        optimizer = torch.optim.Adam(model_density.parameters(),
                                     lr=preset["nn"]["lr"],
                                     weight_decay=preset["nn"]["weight_decay"])
        #
        var_time_0 = time.time()
        var_best_weight = train_density(model=model_density,
                                        optimizer=optimizer,
                                        data_train_set=data_train_set,
                                        data_valid_set=data_valid_set,
                                        var_batch_size=preset["nn"]["batch_size"],
                                        var_epochs=preset["nn"]["epoch"],
                                        device=device,
                                        var_count_loss_weight=preset["density"]["count_loss_weight"])
        var_time_1 = time.time()

        ##
        ## ---------------------------------------- Test ------------------------------------------
        #
        model_density.load_state_dict(var_best_weight)
        model_density.eval()
        test_loader = torch.utils.data.DataLoader(data_test_set,
                                                  batch_size=preset["nn"]["batch_size"], shuffle=False)
        pred_density = []
        with torch.no_grad():
            for var_x, _ in test_loader:
                var_density, _ = model_density(var_x.to(device))
                pred_density.append(var_density.cpu())
        pred_density = torch.cat(pred_density, dim=0).numpy()
        var_time_2 = time.time()
        #
        ## -------------------------------------- Evaluate ----------------------------------------
        #
        var_count_metrics = count_metrics(data_test_y.sum(axis=(1, 2)).round(),
                                          pred_density.sum(axis=(1, 2)).round())
        var_loc_metrics = localization_metrics(data_test_y, pred_density,
                                               var_threshold_frac=preset["density"]["peak_threshold"])
        #
        wandb.log({
            "repeat": var_r,
            "train_time": var_time_1 - var_time_0,
            "test_time": var_time_2 - var_time_1,
            "accuracy": var_count_metrics["accuracy"],
            "mae": var_count_metrics["mae"],
            "occupancy_accuracy": var_count_metrics["occupancy_accuracy"],
            "occupancy_f1": var_count_metrics["occupancy_f1"],
            "loc_error": var_loc_metrics["loc_error"],
            "loc_detection": var_loc_metrics["loc_detection"],
        }, step=var_r + 100000)
        #
        print("  COUNT: Acc %.4f - MAE %.4f - Occupancy Acc %.4f - Occupancy F1 %.4f"
              % (var_count_metrics["accuracy"], var_count_metrics["mae"],
                 var_count_metrics["occupancy_accuracy"], var_count_metrics["occupancy_f1"]))
        print("  WHERE: mean distance %.4f - detection %.4f"
              % (var_loc_metrics["loc_error"], var_loc_metrics["loc_detection"]))
        #
        result_accuracy.append(var_count_metrics["accuracy"])
        result_mae.append(var_count_metrics["mae"])
        result_occ_accuracy.append(var_count_metrics["occupancy_accuracy"])
        result_occ_f1.append(var_count_metrics["occupancy_f1"])
        result_loc_error.append(var_loc_metrics["loc_error"])
        result_loc_detection.append(var_loc_metrics["loc_detection"])
        result_per_class.append(var_count_metrics["per_class_accuracy"])
        #
        if var_r != var_repeat - 1:
            del model_density, optimizer, test_loader, pred_density
            torch.cuda.empty_cache()
            gc.collect()

    #
    ## -------------------------------------- Aggregate ----------------------------------------
    #
    var_num_classes = preset["nn"]["num_count_classes"]

    def mean_se(var_values):
        var_values = np.array(var_values)
        return (float(var_values.mean()),
                float(var_values.std(ddof=1) / np.sqrt(len(var_values))) if len(var_values) > 1 else 0.0)

    results = {}
    for var_name, var_values in (("accuracy", result_accuracy), ("mae", result_mae),
                                 ("occupancy_accuracy", result_occ_accuracy), ("occupancy_f1", result_occ_f1),
                                 ("loc_error", result_loc_error), ("loc_detection", result_loc_detection)):
        var_mean, var_se = mean_se(var_values)
        results[f"avg_{var_name}"] = var_mean
        results[f"se_{var_name}"] = var_se
    results["per_class_accuracy"] = {
        var_class: float(np.mean([var_rep[var_class] for var_rep in result_per_class]))
        for var_class in range(var_num_classes)
    }
    #
    wandb.log({f"avg_{var_name}": results[f"avg_{var_name}"] for var_name in
               ("accuracy", "mae", "occupancy_accuracy", "occupancy_f1", "loc_error", "loc_detection")})
    wandb.finish()
    #
    return results
