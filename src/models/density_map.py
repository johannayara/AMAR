"""
[file]          density_map.py
[description]   Room-agnostic group counting from WiFi CSI via a spatial density map.

                Every person in WiMANS stands at one of the room's 5 known locations, so the model
                predicts a bounded occupancy probability per location and renders the density map as
                a mixture of fixed unit-mass Gaussian kernels placed at those locations (in the
                shared normalized frame defined by preset["layouts"]).

                This keeps the density map well-conditioned:
                  - occupancy is a bounded per-location sigmoid, so it cannot saturate/collapse the
                    way an unbounded softplus count head does;
                  - the count is the integral of the map (sum of occupancies), so count and "where"
                    are consistent by construction;
                  - there is no 1024-way softmax over the grid and no MSE on tiny density values.
"""
#
##

import copy
import gc
import math
import os
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


def build_kernels(var_layout, var_grid_size, var_sigma):
    """
    [description]
    : build the fixed unit-mass Gaussian kernel of each location, in the sorted order of the layout.
    [return]
    : kernels: numpy array (num_locations, grid_size, grid_size)
    : names: list of the location keys, in the same order as the kernels
    """
    #
    var_names = sorted(var_layout)
    var_axis = (np.arange(var_grid_size) + 0.5) / var_grid_size
    var_yy, var_xx = np.meshgrid(var_axis, var_axis, indexing = "ij")  # rows -> y, cols -> x
    #
    var_kernels = []
    for var_name in var_names:
        var_cx, var_cy = var_layout[var_name]
        var_blob = np.exp(-((var_xx - var_cx) ** 2 + (var_yy - var_cy) ** 2) / (2 * var_sigma ** 2))
        var_kernels.append(var_blob / var_blob.sum())
    #
    return np.asarray(var_kernels, dtype = np.float32), var_names


class DensityMapNet(torch.nn.Module):
    """
    [description]
    : CSI backbone + a bounded occupancy head over the room's locations. The density map is the
      occupancy-weighted mixture of the fixed location kernels, so its integral is the predicted
      number of people and its peaks are the predicted locations.
    """
    #
    ##
    def __init__(self,
                 var_x_shape,
                 var_layout,
                 embedding_dim=100,
                 grid_size=32,
                 hidden_dim=256,
                 sigma=0.06):

        super().__init__()
        self.backbone = THAT(var_x_shape, [embedding_dim])
        self.grid_size = grid_size
        #
        var_kernels, self.location_names = build_kernels(var_layout, grid_size, sigma)
        self.register_buffer("kernels", torch.from_numpy(var_kernels))
        #
        self.occupancy_head = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, len(self.location_names)),
        )

    def forward(self, x):
        """
        [return]
        : density: (batch, grid, grid) occupancy-weighted mixture of the location kernels
        : count: (batch,) integral of the density map
        : occupancy_logits: (batch, num_locations) raw occupancy logits
        """
        var_features = self.backbone(x)
        var_occupancy_logits = self.occupancy_head(var_features)
        var_occupancy = torch.sigmoid(var_occupancy_logits)
        #
        var_density = torch.einsum("bl,lhw->bhw", var_occupancy, self.kernels)
        var_count = var_occupancy.sum(-1)
        #
        return var_density, var_count, var_occupancy_logits


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


def predict_counts(var_occupancy, var_threshold=0.5):
    """
    [description]
    : binarize the per-location occupancy at a threshold and count the occupied locations. This is
      the decision rule behind the reported count: a location is occupied when its sigmoid output
      clears the threshold.
    [parameter]
    : var_occupancy: numpy array (N, num_locations) of occupancy probabilities
    : var_threshold: float, occupancy decision threshold
    [return]
    : numpy array (N,) of integer predicted counts
    """
    return (np.asarray(var_occupancy) > var_threshold).sum(axis=1)


def calibrate_threshold(var_occupancy, var_true_count, var_grid=np.linspace(0.05, 0.95, 91)):
    """
    [description]
    : pick the occupancy threshold that maximizes exact-count accuracy on a held-out split. The 0.5
      default is optimal only when the sigmoid outputs are calibrated; sweeping one scalar is a
      cheap post-hoc correction for the mismatch between the count metric and the per-location
      probabilities.
    [parameter]
    : var_occupancy: numpy array (N, num_locations) of occupancy probabilities
    : var_true_count: numpy array (N,) of true counts
    : var_grid: iterable of candidate thresholds
    [return]
    : (threshold, accuracy) at the best threshold
    """
    var_true_count = np.asarray(var_true_count).astype(int)
    var_best_threshold, var_best_accuracy = 0.5, -1.0
    for var_threshold in var_grid:
        var_accuracy = float(np.mean(predict_counts(var_occupancy, var_threshold) == var_true_count))
        ## ties go to the threshold closest to the neutral 0.5, so a small validation split cannot
        ## drag the decision boundary to an extreme
        if (var_accuracy > var_best_accuracy
                or (var_accuracy == var_best_accuracy
                    and abs(var_threshold - 0.5) < abs(var_best_threshold - 0.5))):
            var_best_threshold, var_best_accuracy = float(var_threshold), var_accuracy
    return var_best_threshold, var_best_accuracy


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


def visualize_density_map(data_true_density,
                          data_pred_density,
                          save_dir,
                          var_num_samples=6,
                          var_threshold_frac=0.25,
                          var_tag="density_map"):
    """
    [description]
    : plot the ground-truth and predicted density maps for a spread of test samples (one row per
      sample, ground truth on the left, prediction on the right). Detected locations are circled.
      The title of each panel reports the count, i.e. the integral of the map.
    [return]
    : out_path: str, path of the saved figure
    """
    #
    ##
    os.makedirs(save_dir, exist_ok=True)
    var_num_samples = min(var_num_samples, len(data_true_density))
    #
    ## spread the rows over the whole count range so the figure shows empty, single and crowded frames
    var_true_count = data_true_density.sum(axis=(1, 2))
    var_order = np.argsort(var_true_count)
    var_pick = var_order[np.linspace(0, len(var_order) - 1, var_num_samples).astype(int)]
    #
    var_fig, var_axes = plt.subplots(var_num_samples, 2, figsize=(8, 3 * var_num_samples), squeeze=False)
    var_fig.suptitle(f"Density map - ground truth vs prediction (env {var_tag})")
    #
    for var_row, var_idx in enumerate(var_pick):
        var_maps = ((data_true_density[var_idx], "ground truth"), (data_pred_density[var_idx], "prediction"))
        var_vmax = max(float(data_true_density[var_idx].max()), float(data_pred_density[var_idx].max()), 1e-6)
        #
        for var_col, (var_map, var_title) in enumerate(var_maps):
            var_ax = var_axes[var_row][var_col]
            var_ax.imshow(var_map, origin="upper", cmap="viridis", vmin=0, vmax=var_vmax)
            var_ax.set_xticks([])
            var_ax.set_yticks([])
            #
            var_peaks = _extract_peaks(var_map, var_threshold_frac)
            if len(var_peaks):
                var_ax.scatter(var_peaks[:, 0] * var_map.shape[1] - 0.5,
                               var_peaks[:, 1] * var_map.shape[0] - 0.5,
                               s=70, facecolors="none", edgecolors="red", linewidths=1.5)
            var_ax.set_title(f"{var_title} - count {var_map.sum():.2f}")
    #
    var_fig.tight_layout(rect=[0, 0, 1, 0.95])
    var_out_path = os.path.join(save_dir, f"density_map_{var_tag}.png")
    var_fig.savefig(var_out_path, dpi=120)
    plt.close(var_fig)
    #
    return var_out_path


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
                  patience: int = 150):
    """
    [description]
    : train the density-map model. The objective is the per-location occupancy binary cross-entropy;
      the count is the sum of the occupancies and follows from the same loss. The occupancy threshold
      is calibrated on the validation split each epoch, so model selection tracks the exact-count
      accuracy that is actually reported. The per-epoch validation metric is noisy on a small split,
      so it is smoothed (EMA) before selecting the checkpoint.
    : data_train_set / data_valid_set: TensorDataset of (CSI, occupancy) with occupancy (num_locations,)
      holding 0/1 entries.
    """
    #
    var_train_loader = torch.utils.data.DataLoader(data_train_set, var_batch_size, shuffle=True, pin_memory=True)
    var_valid_loader = torch.utils.data.DataLoader(data_valid_set, len(data_valid_set))
    #
    var_best_score = -np.inf
    var_best_mae = np.inf
    var_best_weight = None
    var_epoch_saved = 0
    var_counter = 0
    ## EMA-smoothed selection scores (0.3 => ~3-epoch memory).
    var_ema_accuracy = None
    var_ema_mae = None
    var_ema_decay = 0.3
    #
    var_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=preset["nn"]["scheduler"]["num_warmup_epochs"] * len(var_train_loader),
        num_training_steps=var_epochs * len(var_train_loader),
        min_lr_ratio=preset["nn"]["scheduler"]["min_lr_ratio"],
    )

    def apply_augmentation(var_x):
        ## The encoder's LayerNorm removes a constant input scale, so the previous +0.1 noise was
        ## negligible against amplitudes around 60 and the gain was largely cancelled. Scale the
        ## noise to the signal and add a small temporal shift; keep the gain and channel dropout.
        var_x = var_x + torch.randn_like(var_x) * (0.05 * var_x.detach().std())
        var_scale = torch.rand(var_x.size(0), 1, device=var_x.device) * 0.2 + 0.9
        var_x = var_x * var_scale.unsqueeze(-1)
        var_shift = int(torch.randint(-var_x.size(1) // 20, var_x.size(1) // 20 + 1, (1,)).item())
        if var_shift != 0:
            var_x = torch.roll(var_x, var_shift, dims=1)
            if var_shift > 0:
                var_x[:, :var_shift] = 0
            else:
                var_x[:, var_shift:] = 0
        var_x = var_x * torch.bernoulli(torch.ones_like(var_x) * 0.96)
        return var_x

    for var_epoch in range(var_epochs):
        var_time_e0 = time.time()
        model.train()
        for var_x, var_occupancy in var_train_loader:
            var_x = apply_augmentation(var_x.to(device))
            var_occupancy = var_occupancy.to(device)
            #
            _, _, var_occupancy_logits = model(var_x)
            var_loss = F.binary_cross_entropy_with_logits(var_occupancy_logits, var_occupancy)
            #
            optimizer.zero_grad()
            var_loss.backward()
            optimizer.step()
            var_scheduler.step()
        #
        model.eval()
        with torch.no_grad():
            var_valid_x, var_valid_y = next(iter(var_valid_loader))
            _, _, var_valid_logits = model(var_valid_x.to(device))
            var_valid_occupancy = torch.sigmoid(var_valid_logits).cpu().numpy()
            var_valid_y = var_valid_y.numpy()
        #
        var_true_count = var_valid_y.sum(axis=1).round()
        var_threshold, _ = calibrate_threshold(var_valid_occupancy, var_true_count)
        var_metrics = count_metrics(var_true_count, predict_counts(var_valid_occupancy, var_threshold))
        #
        ## smooth the selection scores before comparing epochs
        if var_ema_accuracy is None:
            var_ema_accuracy, var_ema_mae = var_metrics["accuracy"], var_metrics["mae"]
        else:
            var_ema_accuracy = var_ema_decay * var_metrics["accuracy"] + (1 - var_ema_decay) * var_ema_accuracy
            var_ema_mae = var_ema_decay * var_metrics["mae"] + (1 - var_ema_decay) * var_ema_mae
        #
        wandb.log({
            "epoch": var_epoch,
            "train_loss": var_loss.item(),
            "valid_accuracy": var_metrics["accuracy"],
            "valid_mae": var_metrics["mae"],
            "valid_occupancy_accuracy": var_metrics["occupancy_accuracy"],
            "valid_occupancy_f1": var_metrics["occupancy_f1"],
            "valid_occupancy_threshold": var_threshold,
            "valid_accuracy_smoothed": var_ema_accuracy,
            "valid_mae_smoothed": var_ema_mae,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        #
        print(f"Epoch {var_epoch}/{var_epochs} - %.3fs" % (time.time() - var_time_e0),
              "- Loss %.6f" % var_loss.cpu(),
              "- Valid Acc %.4f" % var_metrics["accuracy"],
              "- Valid MAE %.4f" % var_metrics["mae"],
              "- Thr %.2f" % var_threshold)
        #
        if (var_ema_accuracy > var_best_score
                or (var_ema_accuracy == var_best_score and var_ema_mae < var_best_mae)):
            var_best_score = var_ema_accuracy
            var_best_mae = var_ema_mae
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
    print(f"Best smoothed exact-count accuracy: {var_best_score:.6f}, smoothed count MAE: {var_best_mae:.6f}")
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
      Counting is read off the per-location occupancy with a validation-calibrated threshold; the
      approximate localization is read off the predicted density map.
    [parameter]
    : data_train_x: numpy array, CSI amplitude to train model
    : data_train_y: numpy array, occupancy targets of shape (N, num_locations) with 0/1 entries
    : data_test_x: numpy array, CSI amplitude to test model
    : data_test_y: numpy array, occupancy targets, same shape as data_train_y
    : var_repeat: int, number of repeated experiments
    : var_task: str, task name kept for interface compatibility (the model is always counting)
    : var_env: str, environment name used for the run name and to pick the location kernels
    : save_path: str, directory for the visualization
    : return: dict, averaged count and localization metrics with SE
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

    var_layout = preset["layouts"][var_env]

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
    var_grid_size = preset["density"]["grid_size"]
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
    var_macs, var_params = get_model_complexity_info(
        DensityMapNet(var_x_shape, var_layout, grid_size=var_grid_size), var_x_shape, as_strings=False)
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
        model_density = DensityMapNet(var_x_shape, var_layout,
                                      embedding_dim=100, grid_size=var_grid_size).to(device)
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
                                        device=device)
        var_time_1 = time.time()

        ##
        ## ---------------------------------------- Test ------------------------------------------
        #
        model_density.load_state_dict(var_best_weight)
        model_density.eval()
        test_loader = torch.utils.data.DataLoader(data_test_set,
                                                  batch_size=preset["nn"]["batch_size"], shuffle=False)
        pred_density, pred_occupancy = [], []
        with torch.no_grad():
            for var_x, _ in test_loader:
                var_density, _, var_logits = model_density(var_x.to(device))
                pred_density.append(var_density.cpu())
                pred_occupancy.append(torch.sigmoid(var_logits).cpu())
        pred_density = torch.cat(pred_density, dim=0).numpy()
        pred_occupancy = torch.cat(pred_occupancy, dim=0).numpy()
        var_time_2 = time.time()
        #
        ## Calibrate the occupancy threshold on the validation split with the selected checkpoint, so
        ## the test count uses the same decision rule that selected the epoch.
        with torch.no_grad():
            _, _, var_valid_logits = model_density(torch.from_numpy(data_valid_x).to(device))
            var_valid_occupancy = torch.sigmoid(var_valid_logits).cpu().numpy()
        var_threshold, _ = calibrate_threshold(var_valid_occupancy, data_valid_y.sum(axis=1).round())
        #
        ## render the ground-truth density from the occupancy targets with the same kernels
        with torch.no_grad():
            ## the kernels are a registered buffer on `device`; the targets are numpy, so move the
            ## kernels to CPU before the einsum
            var_kernels = model_density.kernels.cpu()
            true_density = torch.einsum("bl,lhw->bhw", torch.from_numpy(data_test_y), var_kernels).numpy()
        #
        ## -------------------------------------- Evaluate ----------------------------------------
        #
        var_count_metrics = count_metrics(data_test_y.sum(axis=1).round(),
                                          predict_counts(pred_occupancy, var_threshold))
        var_loc_metrics = localization_metrics(true_density, pred_density,
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
            "occupancy_threshold": var_threshold,
            "loc_error": var_loc_metrics["loc_error"],
            "loc_detection": var_loc_metrics["loc_detection"],
        }, step=var_r + 100000)
        #
        print("  COUNT: Acc %.4f - MAE %.4f - Occupancy Acc %.4f - Occupancy F1 %.4f - Thr %.2f"
              % (var_count_metrics["accuracy"], var_count_metrics["mae"],
                 var_count_metrics["occupancy_accuracy"], var_count_metrics["occupancy_f1"], var_threshold))
        print("  WHERE: mean distance %.4f - detection %.4f"
              % (var_loc_metrics["loc_error"], var_loc_metrics["loc_detection"]))
        #
        if var_r == var_repeat - 1:
            var_fig_path = visualize_density_map(true_density, pred_density, save_path,
                                                 var_threshold_frac=preset["density"]["peak_threshold"],
                                                 var_tag=var_env)
            print(f"  Figure saved to {var_fig_path}")
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
            del model_density, optimizer, test_loader, pred_density, pred_occupancy, true_density
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
