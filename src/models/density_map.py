"""
[file]          density_map.py
[description]   Group counting and localization from WiFi CSI via a continuous spatial density map.

                The model predicts a full H x W occupancy probability map over the shared,
                TX-anchored normalized frame (origin at the transmitter, x right, y away from the
                transmitter; see preset["layouts"]). Unlike the previous formulation, which could only
                place mass on a room's known 5 location kernels, the head is continuous: it can put
                probability at any coordinate, so the same network applies to a room whose layout is
                not known in advance.

                The room's 5 WiMANS locations are still used, but only to render the Gaussian target
                during training and to read the per-location occupancy back off the predicted map at
                evaluation time. The reported count is that occupancy binarized at a calibrated
                threshold (or, in a room with no known locations, the number of map peaks), so count
                and "where" stay consistent by construction.
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
from src.models.losses.supervised_loss import OccupancyDistillationLoss
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


def peak_normalized_kernels(var_kernels):
    """
    [description]
    : rescale each unit-mass kernel to a peak of 1. The target density then holds an occupancy
      probability of ~1 at an occupied location, which is what the per-location readout and the
      threshold calibration expect (a unit-mass kernel peaks at only ~0.04 on a 32x32 grid).
    : var_kernels: tensor (num_locations, grid, grid) or (batch, num_locations, grid, grid)
    : return: same shape as var_kernels
    """
    var_peak = var_kernels.flatten(-2).max(-1).values.clamp_min(1e-12)
    return var_kernels / var_peak.unsqueeze(-1).unsqueeze(-1)


def render_density_targets(var_occupancy, var_kernels):
    """
    [description]
    : render the Gaussian density target of a batch of per-location occupancy vectors. Each occupied
      location contributes one peak-normalized kernel, so the map is a per-cell occupancy
      probability whose peaks are the occupied locations.
    : var_occupancy: tensor (batch, num_locations) with 0/1 entries
    : var_kernels: tensor (num_locations, grid, grid) of unit-mass Gaussian kernels, or a per-sample
      tensor (batch, num_locations, grid, grid) when the batch mixes rooms
    : return: tensor (batch, grid, grid)
    """
    var_peak_kernels = peak_normalized_kernels(var_kernels)
    if var_peak_kernels.dim() == 3:
        return torch.einsum("bl,lhw->bhw", var_occupancy, var_peak_kernels)
    return torch.einsum("bl,blhw->bhw", var_occupancy, var_peak_kernels)


def sample_location_occupancy(var_density, var_kernels):
    """
    [description]
    : read the predicted occupancy at each known location off a continuous density map. The value at
      the location's kernel peak is the natural analogue of the per-location occupancy the previous
      formulation predicted directly, and it keeps the count and per-location metrics unchanged.
    : var_density: tensor (batch, grid, grid) predicted occupancy probabilities
    : var_kernels: tensor (num_locations, grid, grid) of unit-mass Gaussian kernels, or a per-sample
      tensor (batch, num_locations, grid, grid) when the batch mixes rooms
    : return: tensor (batch, num_locations)
    """
    var_density_flat = var_density.flatten(1)
    if var_kernels.dim() == 3:
        return var_density_flat[:, var_kernels.flatten(1).argmax(1)]
    var_peak_index = var_kernels.flatten(2).argmax(2)
    return var_density_flat.gather(1, var_peak_index)


class DensityMapNet(torch.nn.Module):
    """
    [description]
    : CSI backbone + a continuous density head. The head upsamples a seed feature into a full
      grid x grid occupancy probability map over the shared normalized frame, so it can place mass at
      any coordinate rather than only on a room's known locations. The fixed location kernels are kept
      only as a buffer, to render training targets and to read per-location occupancy back off the map.
      The decoder is deliberately small and regularised: a large decoder can satisfy the source-room
      loss by emitting that room's marginal map and ignoring the input, which does not transfer.
    """
    #
    ##
    def __init__(self,
                 var_x_shape,
                 var_layout,
                 embedding_dim=100,
                 grid_size=32,
                 hidden_dim=None,
                 sigma=0.06,
                 dropout=None):

        super().__init__()
        if hidden_dim is None:
            hidden_dim = preset["density"].get("decoder_hidden", 64)
        if dropout is None:
            dropout = preset["density"].get("decoder_dropout", 0.1)
        #
        self.backbone = THAT(var_x_shape, [embedding_dim])
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        #
        var_kernels, self.location_names = build_kernels(var_layout, grid_size, sigma)
        self.register_buffer("kernels", torch.from_numpy(var_kernels))
        #
        ## Seed the map at 4x4 and upsample by 2 three times to 32x32; interpolate if grid_size differs.
        self.decoder_fc = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim * 4 * 4),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout),
        )
        self.decoder_conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(hidden_dim, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(dropout),
            torch.nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """
        [return]
        : density: (batch, grid, grid) predicted occupancy probability map
        : count: (batch,) integral of the density map (soft mass, not thresholded)
        : density_logits: (batch, grid, grid) raw map logits
        """
        var_features = self.backbone(x)
        var_seed = self.decoder_fc(var_features).view(-1, self.hidden_dim, 4, 4)
        var_density_logits = self.decoder_conv(var_seed).squeeze(1)
        if var_density_logits.shape[-1] != self.grid_size:
            var_density_logits = F.interpolate(var_density_logits.unsqueeze(1),
                                               size=(self.grid_size, self.grid_size),
                                               mode="bilinear", align_corners=False).squeeze(1)
        #
        var_density = torch.sigmoid(var_density_logits)
        var_count = var_density.sum(dim=(1, 2))
        #
        return var_density, var_count, var_density_logits


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


def _binary_f1(var_true, var_pred):
    """
    [description]
    : F1 of two boolean arrays, 0.0 when there is nothing positive to score.
    """
    var_true = np.asarray(var_true, dtype=bool)
    var_pred = np.asarray(var_pred, dtype=bool)
    var_tp = float(np.sum(var_true & var_pred))
    var_fp = float(np.sum(~var_true & var_pred))
    var_fn = float(np.sum(var_true & ~var_pred))
    var_precision = var_tp / (var_tp + var_fp) if (var_tp + var_fp) > 0 else 0.0
    var_recall = var_tp / (var_tp + var_fn) if (var_tp + var_fn) > 0 else 0.0
    return (2 * var_precision * var_recall / (var_precision + var_recall)
            if (var_precision + var_recall) > 0 else 0.0)


def count_metrics(var_true_count, var_pred_count,
                  var_true_occupancy=None, var_pred_occupancy=None,
                  var_threshold=0.5, var_num_classes=6):
    """
    [description]
    : group-count metrics: exact-count accuracy, count MAE, per-location occupancy accuracy/F1 and
      per-count accuracy.
    : var_true_occupancy / var_pred_occupancy: optional (N, num_locations) occupancy targets and
      predicted probabilities. When supplied, the occupancy metrics are computed over every
      (sample, location) cell with var_pred_occupancy binarized at var_threshold. Room-level presence
      ("is anyone in the room") is deliberately not reported: 94.7% of WiMANS frames contain someone,
      so always answering "yes" already scores 0.947 accuracy / 0.973 F1 and the metric hides every
      per-location error.
    : var_threshold: occupancy decision threshold used to binarize var_pred_occupancy.
    """
    #
    var_true_count = np.asarray(var_true_count).astype(int)
    var_pred_count = np.asarray(var_pred_count).astype(int)
    #
    var_exact = float(np.mean(var_true_count == var_pred_count))
    var_mae = float(np.mean(np.abs(var_true_count - var_pred_count)))
    #
    ## per-location occupancy: which of the room's locations are occupied
    if var_true_occupancy is not None and var_pred_occupancy is not None:
        var_true_occ = np.asarray(var_true_occupancy) > 0.5
        var_pred_occ = np.asarray(var_pred_occupancy) > var_threshold
        var_occ_acc = float(np.mean(var_true_occ == var_pred_occ))
        var_occ_f1 = _binary_f1(var_true_occ, var_pred_occ)
    else:
        var_occ_acc, var_occ_f1 = float("nan"), float("nan")
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


def visualize_density_map(data_true_density,
                          data_pred_density,
                          save_dir,
                          var_num_samples=6,
                          var_threshold_frac=0.25,
                          var_tag="density_map"):
    """
    [description]
    : plot the ground-truth and predicted density maps for a spread of test samples (one row per
      sample, ground truth on the left, prediction on the right). Detected locations are circled and
      the title of each panel reports the number of detected peaks.
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
            var_ax.set_title(f"{var_title} - {len(var_peaks)} peaks")
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
def count_class_sampling_weights(data_set):
    """
    [description]
    : per-sample sampling weight that equalizes the count classes (0..num_count_classes-1), so the
      rare empty-room (count-0) frames are drawn as often as every other class. WiMANS has only 5.3%
      count-0 frames per room (99 of 1881), so a plain shuffle lets the model pay almost no price for
      never predicting an empty room. Inverse-frequency weighting fixes that without discarding any
      data, unlike undersampling the non-empty classes.
    : data_set: TensorDataset of (CSI, occupancy), or a Subset / ConcatDataset of one.
    : return: numpy array (N,) of non-negative sampling weights
    """
    var_count = _dataset_count_classes(data_set)
    var_num_classes = preset["nn"]["num_count_classes"]
    var_freq = np.bincount(var_count, minlength=var_num_classes).astype(np.float64)
    var_inverse = np.where(var_freq > 0, 1.0 / np.maximum(var_freq, 1.0), 0.0)
    #
    return var_inverse[var_count]


def _dataset_count_classes(data_set):
    """
    [description]
    : count class (0..5) of every sample in a TensorDataset, Subset or ConcatDataset of them.
    : return: numpy array (N,) of ints
    """
    if isinstance(data_set, torch.utils.data.ConcatDataset):
        return np.concatenate([_dataset_count_classes(var_part) for var_part in data_set.datasets])
    if isinstance(data_set, torch.utils.data.Subset):
        return _dataset_count_classes(data_set.dataset)[data_set.indices]
    return data_set.tensors[1].sum(axis=1).round().long().numpy()


def save_density_checkpoint(model, var_x_shape, var_env, save_path, var_tag="model"):
    """
    [description]
    : save a trained density model together with the metadata needed to rebuild it, so a live CSI
      capture can be run through it later (scripts/run_pcap_inference.py).
    : return: str, path of the saved checkpoint
    """
    os.makedirs(save_path, exist_ok=True)
    var_out_path = os.path.join(save_path, f"{var_tag}.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "x_shape": tuple(var_x_shape),
        "layout_name": var_env,
        "grid_size": model.grid_size,
        "sigma": preset["density"]["sigma"],
        "decoder_hidden": model.hidden_dim,
        "decoder_dropout": model.dropout,
        "location_names": list(model.location_names),
    }, var_out_path)
    print(f"  Checkpoint saved to {var_out_path}")
    return var_out_path


def train_density(model,
                  optimizer,
                  data_train_set: TensorDataset,
                  data_valid_set: TensorDataset,
                  var_batch_size: int,
                  var_epochs: int,
                  device,
                  patience: int = 150,
                  teacher=None,
                  kd_loss=None,
                  kd_weight: float = 0.0,
                  var_kernel_bank=None):
    """
    [description]
    : train the density-map model. The objective is the binary cross-entropy between the predicted
      density map and the Gaussian target rendered from the occupancy labels; the count and the
      per-location occupancy are read back off the map. The occupancy threshold is calibrated on the
      validation split each epoch, so model selection tracks the exact-count accuracy that is
      actually reported. The per-epoch validation metric is noisy on a small split, so it is smoothed
      (EMA) before selecting the checkpoint. When preset["density"]["balance_empty_class"] is set,
      the training batches are drawn with class-balanced weights so the under-represented empty-room
      frames are seen as often as every other count class. When a frozen teacher and a kd_loss are
      given, the teacher's density logits are distilled into the student with weight kd_weight
      (few-shot knowledge distillation).
    : data_train_set / data_valid_set: TensorDataset of (CSI, occupancy) with occupancy (num_locations,)
      holding 0/1 entries. The density target is rendered from the occupancy with the model's kernels.
      When the set mixes several rooms it yields a third element, the room index, and var_kernel_bank
      must be given so each sample is rendered and read out with its own room's kernels.
    : var_kernel_bank: optional tensor (num_rooms, num_locations, grid, grid) of unit-mass kernels,
      one entry per room index carried by the datasets.
    : teacher: optional frozen DensityMapNet whose density logits supervise the student.
    : kd_loss: optional distillation criterion taking (student_outputs, teacher_outputs).
    : kd_weight: weight of the distillation term.
    """
    #
    if preset["density"].get("balance_empty_class", False):
        ## Draw every count class equally often: the empty-room frames are only 5.3% of WiMANS, so
        ## without this the model rarely pays for never predicting an empty room.
        var_sample_weights = count_class_sampling_weights(data_train_set)
        var_sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(var_sample_weights, dtype=torch.double),
            num_samples=len(data_train_set), replacement=True)
        var_train_loader = torch.utils.data.DataLoader(
            data_train_set, var_batch_size, sampler=var_sampler, pin_memory=True)
    else:
        var_train_loader = torch.utils.data.DataLoader(
            data_train_set, var_batch_size, shuffle=True, pin_memory=True)
    var_valid_loader = torch.utils.data.DataLoader(data_valid_set, len(data_valid_set))
    #
    var_use_distillation = teacher is not None and kd_loss is not None and kd_weight > 0.0
    var_kd_value = 0.0
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
        for var_batch in var_train_loader:
            var_x, var_occupancy = var_batch[0], var_batch[1]
            var_x = apply_augmentation(var_x.to(device))
            var_occupancy = var_occupancy.to(device)
            #
            ## each sample is rendered with its own room's kernels when the batch mixes rooms
            if len(var_batch) > 2:
                var_kernels = var_kernel_bank[var_batch[2].to(device)]
            else:
                var_kernels = model.kernels
            #
            _, _, var_density_logits = model(var_x)
            var_density_target = render_density_targets(var_occupancy, var_kernels)
            var_loss = F.binary_cross_entropy_with_logits(var_density_logits, var_density_target)
            #
            ## few-shot distillation: the frozen teacher supervises the student's density logits
            var_kd_value = 0.0
            if var_use_distillation:
                with torch.no_grad():
                    _, _, var_teacher_logits = teacher(var_x)
                var_kd = kd_loss(var_density_logits, var_teacher_logits)
                var_kd_value = float(var_kd.detach())
                var_loss = var_loss + kd_weight * var_kd
            #
            optimizer.zero_grad()
            var_loss.backward()
            optimizer.step()
            var_scheduler.step()
        #
        model.eval()
        with torch.no_grad():
            var_valid_batch = next(iter(var_valid_loader))
            var_valid_x, var_valid_y = var_valid_batch[0], var_valid_batch[1]
            var_valid_density, _, var_valid_logits = model(var_valid_x.to(device))
            var_valid_kd_value = 0.0
            if var_use_distillation:
                _, _, var_valid_teacher_logits = teacher(var_valid_x.to(device))
                var_valid_kd_value = float(kd_loss(var_valid_logits, var_valid_teacher_logits).detach())
            if len(var_valid_batch) > 2:
                var_valid_kernels = var_kernel_bank[var_valid_batch[2].to(device)]
            else:
                var_valid_kernels = model.kernels
            var_valid_occupancy = sample_location_occupancy(
                var_valid_density, var_valid_kernels).cpu().numpy()
            var_valid_y = var_valid_y.numpy()
        #
        var_true_count = var_valid_y.sum(axis=1).round()
        var_threshold, _ = calibrate_threshold(var_valid_occupancy, var_true_count)
        var_metrics = count_metrics(var_true_count, predict_counts(var_valid_occupancy, var_threshold),
                                    var_valid_y, var_valid_occupancy, var_threshold)
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
            "train_kd": var_kd_value,
            "valid_kd": var_valid_kd_value,
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
      Counting is read off the per-location occupancy with a validation-calibrated threshold.
    [parameter]
    : data_train_x: numpy array, CSI amplitude to train model
    : data_train_y: numpy array, occupancy targets of shape (N, num_locations) with 0/1 entries
    : data_test_x: numpy array, CSI amplitude to test model
    : data_test_y: numpy array, occupancy targets, same shape as data_train_y
    : var_repeat: int, number of repeated experiments
    : var_task: str, task name kept for interface compatibility (the model is always counting)
    : var_env: str, environment name used for the run name and to pick the location kernels
    : save_path: str, directory for the visualization
    : return: dict, averaged count and per-location occupancy metrics with SE
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
        pred_density = []
        with torch.no_grad():
            for var_x, _ in test_loader:
                var_density, _, _ = model_density(var_x.to(device))
                pred_density.append(var_density.cpu())
        pred_density = torch.cat(pred_density, dim=0)
        ## read the per-location occupancy off the continuous map at this room's known locations
        pred_occupancy = sample_location_occupancy(pred_density, model_density.kernels.cpu()).numpy()
        pred_density = pred_density.numpy()
        var_time_2 = time.time()
        #
        ## Calibrate the occupancy threshold on the validation split with the selected checkpoint, so
        ## the test count uses the same decision rule that selected the epoch.
        with torch.no_grad():
            var_valid_density, _, _ = model_density(torch.from_numpy(data_valid_x).to(device))
            var_valid_occupancy = sample_location_occupancy(
                var_valid_density, model_density.kernels).cpu().numpy()
        var_threshold, _ = calibrate_threshold(var_valid_occupancy, data_valid_y.sum(axis=1).round())
        #
        ## render the ground-truth density from the occupancy targets with the same kernels
        with torch.no_grad():
            ## the kernels are a registered buffer on `device`; the targets are numpy, so move the
            ## kernels to CPU before the einsum
            var_kernels = peak_normalized_kernels(model_density.kernels.cpu())
            true_density = torch.einsum("bl,lhw->bhw", torch.from_numpy(data_test_y), var_kernels).numpy()
        #
        ## -------------------------------------- Evaluate ----------------------------------------
        #
        var_count_metrics = count_metrics(data_test_y.sum(axis=1).round(),
                                          predict_counts(pred_occupancy, var_threshold),
                                          data_test_y, pred_occupancy, var_threshold)
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
        }, step=var_r + 100000)
        #
        print("  COUNT: Acc %.4f - MAE %.4f - Occupancy Acc %.4f - Occupancy F1 %.4f - Thr %.2f"
              % (var_count_metrics["accuracy"], var_count_metrics["mae"],
                 var_count_metrics["occupancy_accuracy"], var_count_metrics["occupancy_f1"], var_threshold))
        #
        if var_r == var_repeat - 1:
            var_fig_path = visualize_density_map(true_density, pred_density, save_path,
                                                 var_threshold_frac=preset["density"]["peak_threshold"],
                                                 var_tag=var_env)
            ## the standard per-location performance figures, on the binarized occupancy
            visualize_model_performance(pred_occupancy, data_test_y, save_dir=save_path,
                                        var_mode="occupancy", var_threshold=var_threshold)
            ## keep the last repeat's weights so a live capture can be run through the model
            save_density_checkpoint(model_density, var_x_shape, var_env, save_path)
            print(f"  Figure saved to {var_fig_path}")
        #
        result_accuracy.append(var_count_metrics["accuracy"])
        result_mae.append(var_count_metrics["mae"])
        result_occ_accuracy.append(var_count_metrics["occupancy_accuracy"])
        result_occ_f1.append(var_count_metrics["occupancy_f1"])
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
                                 ("occupancy_accuracy", result_occ_accuracy), ("occupancy_f1", result_occ_f1)):
        var_mean, var_se = mean_se(var_values)
        results[f"avg_{var_name}"] = var_mean
        results[f"se_{var_name}"] = var_se
    results["per_class_accuracy"] = {
        var_class: float(np.mean([var_rep[var_class] for var_rep in result_per_class]))
        for var_class in range(var_num_classes)
    }
    #
    wandb.log({f"avg_{var_name}": results[f"avg_{var_name}"] for var_name in
               ("accuracy", "mae", "occupancy_accuracy", "occupancy_f1")})
    wandb.finish()
    #
    return results


#
## ---------------------------------------------------------------------------------------------- ##
## ------------------------------------ cross-domain runner -------------------------------------- ##
#
##
def run_density_map_cross_domain(train_sets_by_env,
                                 test_sets_by_env,
                                 var_repeat=10, var_task="count",
                                 save_path="./visualizations/temp"):
    """
    [description]
    : cross-domain run of the density-map group-counting model. Trains on every room in
      train_sets_by_env and evaluates on every room in test_sets_by_env. Pass a single training room
      for the old one-room protocol, or two training rooms for leave-one-room-out (train on two, test
      on the third), which is what stops the decoder from memorising one room's spatial layout.
      Each training room contributes its own 90/10 split, so model selection and the occupancy
      threshold are anchored on a validation split that covers every training room; the test rooms
      contribute no labels at all.
    [parameter]
    : train_sets_by_env: dict, env name -> (CSI amplitude, occupancy targets) of a training room
    : test_sets_by_env: dict, env name -> (CSI amplitude, occupancy targets) of a held-out room
    : var_repeat: int, number of repeated experiments
    : var_task: str, task name kept for interface compatibility (the model is always counting)
    : save_path: str, directory for the per-room visualizations
    : return: dict, env name -> averaged count and per-location occupancy metrics with SE
    """
    #
    device = select_device()
    print(f"Using device: {device}")
    #
    for var_env_name in list(test_sets_by_env):
        var_x_env, var_y_env = test_sets_by_env[var_env_name]
        test_sets_by_env[var_env_name] = (var_x_env, np.asarray(var_y_env, dtype=np.float32))
    #
    var_train_envs = list(train_sets_by_env)
    var_layout = preset["layouts"][var_train_envs[0]]
    var_grid_size = preset["density"]["grid_size"]
    var_sigma = preset["density"]["sigma"]
    #
    ## one unit-mass kernel bank per training room; each sample is rendered and read out with its own
    ## room's kernels, which is what lets several rooms share one model
    var_kernel_bank = torch.stack([
        torch.from_numpy(build_kernels(preset["layouts"][var_env_name], var_grid_size, var_sigma)[0])
        for var_env_name in var_train_envs])
    var_room_index = {var_env_name: var_idx for var_idx, var_env_name in enumerate(var_train_envs)}

    #
    ## ============================================ Preprocess ============================================
    #
    ## Each training room keeps its CSI in place: torch.from_numpy shares the buffer and the 90/10
    ## split is a Subset over indices, so a multi-room run costs no extra CSI memory beyond the rooms
    ## themselves (concatenating the arrays would copy several GB per room and OOM the node).
    var_train_datasets, var_valid_datasets = [], []
    var_valid_x, var_valid_y, var_valid_room = [], [], []
    for var_env_name in var_train_envs:
        var_x_env, var_y_env = train_sets_by_env[var_env_name]
        var_y_env = np.asarray(var_y_env, dtype=np.float32)
        var_x_env = np.asarray(var_x_env).reshape(var_x_env.shape[0], var_x_env.shape[1], -1)
        var_room_env = np.full(var_x_env.shape[0], var_room_index[var_env_name], dtype=np.int64)
        var_dataset = TensorDataset(torch.from_numpy(var_x_env), torch.from_numpy(var_y_env),
                                    torch.from_numpy(var_room_env))
        #
        var_perm = np.random.RandomState(39).permutation(len(var_dataset))
        var_num_valid = max(1, int(round(0.1 * len(var_dataset))))
        var_valid_idx = np.sort(var_perm[:var_num_valid])
        var_train_idx = np.sort(var_perm[var_num_valid:])
        var_train_datasets.append(torch.utils.data.Subset(var_dataset, var_train_idx.tolist()))
        var_valid_datasets.append(torch.utils.data.Subset(var_dataset, var_valid_idx.tolist()))
        ## only the 10% validation slice is copied; it anchors selection and the threshold
        var_valid_x.append(var_x_env[var_valid_idx])
        var_valid_y.append(var_y_env[var_valid_idx])
        var_valid_room.append(var_room_env[var_valid_idx])
    #
    data_train_set = torch.utils.data.ConcatDataset(var_train_datasets)
    data_valid_set = torch.utils.data.ConcatDataset(var_valid_datasets)
    var_valid_x = np.concatenate(var_valid_x)
    var_valid_y = np.concatenate(var_valid_y)
    var_valid_room = np.concatenate(var_valid_room)
    #
    var_x_shape = tuple(var_train_datasets[0].dataset.tensors[0].shape[1:])
    print(f"Training rooms: {var_train_envs} ({len(data_train_set)} train / "
          f"{len(data_valid_set)} valid) | Test rooms: {list(test_sets_by_env)}")
    #
    var_macs, var_params = get_model_complexity_info(
        DensityMapNet(var_x_shape, var_layout, grid_size=var_grid_size), var_x_shape, as_strings=False)
    print("Parameters:", var_params, "- FLOPs:", var_macs * 2)

    #
    ## ========================================= Train & Evaluate =========================================
    #
    env_rep_metrics = {}
    env_last_density = {}
    env_last_occupancy = {}
    #
    for var_r in range(var_repeat):
        print("Repeat", var_r)
        name_run = f"DensityMapCD{var_r}_" + "_".join(var_train_envs)
        wandb.init(project="density_map_cross_domain", name=name_run, config=preset, reinit=True)
        #
        torch.random.manual_seed(var_r + 39)
        #
        model_density = DensityMapNet(var_x_shape, var_layout,
                                      embedding_dim=100, grid_size=var_grid_size).to(device)
        optimizer = torch.optim.Adam(model_density.parameters(),
                                     lr=preset["nn"]["lr"],
                                     weight_decay=preset["nn"]["weight_decay"])
        #
        var_best_weight = train_density(model=model_density,
                                        optimizer=optimizer,
                                        data_train_set=data_train_set,
                                        data_valid_set=data_valid_set,
                                        var_batch_size=preset["nn"]["batch_size"],
                                        var_epochs=preset["nn"]["epoch"],
                                        device=device,
                                        var_kernel_bank=var_kernel_bank)
        model_density.load_state_dict(var_best_weight)
        model_density.eval()
        #
        ## occupancy threshold from the training rooms' validation split
        with torch.no_grad():
            var_valid_density, _, _ = model_density(torch.from_numpy(var_valid_x).to(device))
            var_valid_occupancy = sample_location_occupancy(
                var_valid_density,
                var_kernel_bank[torch.from_numpy(var_valid_room).to(device)]).cpu().numpy()
        var_threshold, _ = calibrate_threshold(var_valid_occupancy, var_valid_y.sum(axis=1).round())
        #
        ## -------------------------------------- Test per room ----------------------------------------
        #
        for var_env_name, (var_x_env, var_y_env) in test_sets_by_env.items():
            var_x_env = var_x_env.reshape(var_x_env.shape[0], var_x_env.shape[1], -1)
            ## render the maps with the test room's own layout
            var_test_kernels = torch.from_numpy(
                build_kernels(preset["layouts"][var_env_name], var_grid_size, var_sigma)[0])
            env_loader = torch.utils.data.DataLoader(
                TensorDataset(torch.from_numpy(var_x_env)),
                batch_size=preset["nn"]["batch_size"], shuffle=False)
            var_density_pred = []
            with torch.no_grad():
                for (var_x,) in env_loader:
                    var_density, _, _ = model_density(var_x.to(device))
                    var_density_pred.append(var_density.cpu())
            pred_density = torch.cat(var_density_pred, dim=0)
            ## the map is continuous; read this test room's occupancy off its own known locations
            var_occupancy = sample_location_occupancy(pred_density, var_test_kernels).numpy()
            true_density = torch.einsum("bl,lhw->bhw",
                                        torch.from_numpy(var_y_env),
                                        peak_normalized_kernels(var_test_kernels)).numpy()
            pred_density = pred_density.numpy()
            #
            var_count_metrics = count_metrics(var_y_env.sum(axis=1).round(),
                                              predict_counts(var_occupancy, var_threshold),
                                              var_y_env, var_occupancy, var_threshold)
            #
            var_rep = {**var_count_metrics, "threshold": var_threshold}
            env_rep_metrics.setdefault(var_env_name, []).append(var_rep)
            env_last_density[var_env_name] = (true_density, pred_density)
            env_last_occupancy[var_env_name] = (var_occupancy, var_threshold, var_y_env)
            #
            wandb.log({
                f"test_results_per_env/{var_env_name}/accuracy": var_count_metrics["accuracy"],
                f"test_results_per_env/{var_env_name}/mae": var_count_metrics["mae"],
                f"test_results_per_env/{var_env_name}/occupancy_accuracy": var_count_metrics["occupancy_accuracy"],
                f"test_results_per_env/{var_env_name}/occupancy_f1": var_count_metrics["occupancy_f1"],
                f"test_results_per_env/{var_env_name}/threshold": var_threshold,
            }, step=var_r + 100000)
            #
            print(f"  [{var_env_name}] COUNT Acc {var_count_metrics['accuracy']:.4f} - "
                  f"MAE {var_count_metrics['mae']:.4f} - "
                  f"Occ Acc {var_count_metrics['occupancy_accuracy']:.4f} - "
                  f"Occ F1 {var_count_metrics['occupancy_f1']:.4f} (Thr {var_threshold:.2f})")
        #
        if var_r != var_repeat - 1:
            del model_density, optimizer
            torch.cuda.empty_cache()
            gc.collect()
        else:
            ## keep the last repeat's weights so a live capture can be run through the model
            save_density_checkpoint(model_density, var_x_shape, var_train_envs[0], save_path)

    #
    ## -------------------------------------- Aggregate per room ----------------------------------------
    #
    var_num_classes = preset["nn"]["num_count_classes"]
    var_metric_names = ("accuracy", "mae", "occupancy_accuracy", "occupancy_f1")
    results = {}
    for var_env_name, var_rep_list in env_rep_metrics.items():
        var_env_result = {}
        for var_name in var_metric_names:
            var_arr = np.array([var_rep[var_name] for var_rep in var_rep_list])
            var_env_result[f"avg_{var_name}"] = float(var_arr.mean())
            var_std = float(var_arr.std(ddof=1)) if len(var_arr) > 1 else 0.0
            var_env_result[f"std_{var_name}"] = var_std
            var_env_result[f"se_{var_name}"] = var_std / np.sqrt(len(var_arr)) if len(var_arr) > 1 else 0.0
        var_env_result["avg_threshold"] = float(np.mean([var_rep["threshold"] for var_rep in var_rep_list]))
        var_env_result["per_class_accuracy"] = {
            var_class: float(np.mean([var_rep["per_class_accuracy"][var_class] for var_rep in var_rep_list]))
            for var_class in range(var_num_classes)
        }
        results[var_env_name] = var_env_result
        #
        var_true_density, var_pred_density = env_last_density[var_env_name]
        visualize_density_map(var_true_density, var_pred_density,
                              os.path.join(save_path, var_env_name),
                              var_threshold_frac=preset["density"]["peak_threshold"],
                              var_tag=var_env_name)
        ## the standard per-location performance figures, on the binarized occupancy of the last repeat
        var_occupancy_last, var_threshold_last, var_y_last = env_last_occupancy[var_env_name]
        visualize_model_performance(var_occupancy_last, var_y_last,
                                    save_dir=os.path.join(save_path, var_env_name),
                                    var_mode="occupancy", var_threshold=var_threshold_last)
        #
        print(f"\n[{var_env_name}] avg over {var_repeat} repeats: "
              f"Accuracy {var_env_result['avg_accuracy']:.4f} ± {var_env_result['se_accuracy']:.4f} | "
              f"MAE {var_env_result['avg_mae']:.4f} ± {var_env_result['se_mae']:.4f} | "
              f"Occ Acc {var_env_result['avg_occupancy_accuracy']:.4f} ± {var_env_result['se_occupancy_accuracy']:.4f} | "
              f"Occ F1 {var_env_result['avg_occupancy_f1']:.4f} ± {var_env_result['se_occupancy_f1']:.4f}")
    #
    wandb.finish()
    #
    return results


#
## ---------------------------------------------------------------------------------------------- ##
## ------------------------------------ few-shot runner ------------------------------------------ ##
#
##
def run_density_map_few_shot(data_train_x,
                             data_train_y,
                             test_sets_by_env,
                             var_few_shot_ratio=0.05,
                             var_kd_weight=1.0,
                             var_kd_temperature=1.0,
                             var_teacher_epochs=None,
                             var_student_epochs=None,
                             var_compile=True,
                             var_repeat=10, var_task="count", var_env="empty_room",
                             save_path="./visualizations/temp"):
    """
    [description]
    : Few-shot knowledge distillation for the density-map group-counting model, mirroring
      run_AMAR_WO_RVQ_few_shot. A teacher DensityMapNet is trained on the full training environment; a
      student DensityMapNet is trained on a small fraction (var_few_shot_ratio) of that same
      environment with the supervised occupancy loss plus kd_weight * OccupancyDistillationLoss
      against the frozen teacher. Both are evaluated on every other environment in test_sets_by_env.
      Because the density model is a per-location sigmoid predictor (not a set predictor), the soft
      targets are matched element-wise with a temperature-scaled Bernoulli cross-entropy: no
      Hungarian assignment is involved.
      The protocol deliberately mirrors the AMAR few-shot runner (same 39-seed split, same validation
      slice), so the two models' few-shot numbers are directly comparable.
    [parameter]
    : data_train_x: numpy array, CSI amplitude of the single training environment
    : data_train_y: numpy array, per-location occupancy targets (N, num_locations) with 0/1 entries
    : test_sets_by_env: dict, {env_name: (X, y)} test sets of the other environments, y occupancy
    : var_few_shot_ratio: float, fraction of the training environment used to train the student
    : var_kd_weight: float, weight of the distillation term
    : var_kd_temperature: float, temperature of the distillation soft targets
    : var_teacher_epochs: int, teacher training epochs (defaults to preset["nn"]["epoch"])
    : var_student_epochs: int, student training epochs (defaults to preset["nn"]["epoch"])
    : var_compile: bool, torch.compile the backbones of both models
    : var_repeat: int, number of repeated experiments
    : var_env: str or list, training environment name(s) used for the run name and the teacher layout
    : save_path: str, directory for visualizations (one sub-directory per test environment)
    : return: dict, per-environment averaged count and per-location occupancy metrics with SE
    """
    #
    device = select_device()
    print(f"Using device: {device}")

    if var_teacher_epochs is None:
        var_teacher_epochs = preset["nn"]["epoch"]
    if var_student_epochs is None:
        var_student_epochs = preset["nn"]["epoch"]
    env_name = var_env if isinstance(var_env, str) else "_".join(var_env)

    #
    ## ============================================ Preprocess ============================================
    #
    data_train_y = np.asarray(data_train_y, dtype=np.float32)
    for var_env_key in list(test_sets_by_env):
        var_x_env, var_y_env = test_sets_by_env[var_env_key]
        test_sets_by_env[var_env_key] = (var_x_env, np.asarray(var_y_env, dtype=np.float32))

    var_layout = preset["layouts"][var_env]
    var_grid_size = preset["density"]["grid_size"]
    var_sigma = preset["density"]["sigma"]

    data_train_x = data_train_x.reshape(data_train_x.shape[0], data_train_x.shape[1], -1)
    var_x_shape = data_train_x[0].shape

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

    data_few_x, data_few_y = data_train_x[few_idx], data_train_y[few_idx]
    data_teacher_valid_x, data_teacher_valid_y = data_train_x[valid_idx], data_train_y[valid_idx]
    data_teacher_x, data_teacher_y = data_train_x[teacher_idx], data_train_y[teacher_idx]

    teacher_train_set = TensorDataset(torch.from_numpy(data_teacher_x), torch.from_numpy(data_teacher_y))
    teacher_valid_set = TensorDataset(torch.from_numpy(data_teacher_valid_x), torch.from_numpy(data_teacher_valid_y))
    student_train_set = TensorDataset(torch.from_numpy(data_few_x), torch.from_numpy(data_few_y))
    student_valid_set = TensorDataset(torch.from_numpy(data_teacher_valid_x), torch.from_numpy(data_teacher_valid_y))
    print(f"Training environment [{env_name}] - teacher train {data_teacher_x.shape[0]} | "
          f"student few-shot train {num_few}/{num_train} ({var_few_shot_ratio:.2%}) | "
          f"student validation {data_teacher_valid_x.shape[0]}")
    print(f"Test environments: {list(test_sets_by_env.keys())}")

    #
    ## ---------------------------------------- Complexity ----------------------------------------
    #
    var_macs, var_params = get_model_complexity_info(
        DensityMapNet(var_x_shape, var_layout, grid_size=var_grid_size), var_x_shape, as_strings=False)
    print("Parameters:", var_params, "- FLOPs:", var_macs * 2)

    #
    ## ========================================= Train & Evaluate =========================================
    #
    env_rep_metrics = {}
    env_last_density = {}
    env_last_occupancy = {}

    for var_r in range(var_repeat):
        print("Repeat", var_r)
        name_run = f"DensityMapFewShot{var_r}_{env_name}_k{var_few_shot_ratio}"
        wandb.init(project="density_map_few_shot", name=name_run, config=preset, reinit=True)
        #
        torch.random.manual_seed(var_r + 39)
        #
        ## ---------------------------------------- Teacher ----------------------------------------
        #
        teacher = DensityMapNet(var_x_shape, var_layout,
                                embedding_dim=100, grid_size=var_grid_size).to(device)
        if var_compile:
            teacher.backbone = torch.compile(teacher.backbone)
        teacher_optimizer = torch.optim.Adam(teacher.parameters(),
                                             lr=preset["nn"]["lr"],
                                             weight_decay=preset["nn"]["weight_decay"])
        teacher_time_0 = time.time()
        teacher_best_weight = train_density(model=teacher,
                                            optimizer=teacher_optimizer,
                                            data_train_set=teacher_train_set,
                                            data_valid_set=teacher_valid_set,
                                            var_batch_size=preset["nn"]["batch_size"],
                                            var_epochs=var_teacher_epochs,
                                            device=device)
        teacher_time_1 = time.time()
        teacher.load_state_dict(teacher_best_weight)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        #
        ## ---------------------------------------- Student ----------------------------------------
        #
        student = DensityMapNet(var_x_shape, var_layout,
                                embedding_dim=100, grid_size=var_grid_size).to(device)
        if var_compile:
            student.backbone = torch.compile(student.backbone)
        student_optimizer = torch.optim.Adam(student.parameters(),
                                             lr=preset["nn"]["lr"],
                                             weight_decay=preset["nn"]["weight_decay"])
        kd_loss = OccupancyDistillationLoss(temperature=var_kd_temperature)
        student_time_0 = time.time()
        student_best_weight = train_density(model=student,
                                            optimizer=student_optimizer,
                                            data_train_set=student_train_set,
                                            data_valid_set=student_valid_set,
                                            var_batch_size=preset["nn"]["batch_size"],
                                            var_epochs=var_student_epochs,
                                            device=device,
                                            teacher=teacher,
                                            kd_loss=kd_loss,
                                            kd_weight=var_kd_weight)
        student_time_1 = time.time()
        student.load_state_dict(student_best_weight)
        student.eval()

        #
        ## occupancy threshold from the training room's validation split, reused for every test room
        # (no test-room labels are used anywhere)
        #
        with torch.no_grad():
            var_valid_density, _, _ = student(torch.from_numpy(data_teacher_valid_x).to(device))
            var_valid_occupancy = sample_location_occupancy(
                var_valid_density, student.kernels).cpu().numpy()
        var_threshold, _ = calibrate_threshold(var_valid_occupancy, data_teacher_valid_y.sum(axis=1).round())

        #
        ## ---------------------------- Test on the other environments ----------------------------
        #
        for var_env_name, (var_x_env, var_y_env) in test_sets_by_env.items():
            var_x_env = var_x_env.reshape(var_x_env.shape[0], var_x_env.shape[1], -1)
            ## render the maps with the test room's own layout
            var_test_kernels = torch.from_numpy(
                build_kernels(preset["layouts"][var_env_name], var_grid_size, var_sigma)[0])
            env_loader = torch.utils.data.DataLoader(
                TensorDataset(torch.from_numpy(var_x_env)),
                batch_size=preset["nn"]["batch_size"], shuffle=False)
            var_density_pred = []
            with torch.no_grad():
                for (var_x,) in env_loader:
                    var_density, _, _ = student(var_x.to(device))
                    var_density_pred.append(var_density.cpu())
            pred_density = torch.cat(var_density_pred, dim=0)
            ## the map is continuous; read this test room's occupancy off its own known locations
            var_occupancy = sample_location_occupancy(pred_density, var_test_kernels).numpy()
            true_density = torch.einsum("bl,lhw->bhw",
                                        torch.from_numpy(var_y_env),
                                        peak_normalized_kernels(var_test_kernels)).numpy()
            pred_density = pred_density.numpy()
            #
            var_count_metrics = count_metrics(var_y_env.sum(axis=1).round(),
                                              predict_counts(var_occupancy, var_threshold),
                                              var_y_env, var_occupancy, var_threshold)
            #
            var_rep = {**var_count_metrics, "threshold": var_threshold,
                       "teacher_train_time": teacher_time_1 - teacher_time_0,
                       "student_train_time": student_time_1 - student_time_0}
            env_rep_metrics.setdefault(var_env_name, []).append(var_rep)
            env_last_density[var_env_name] = (true_density, pred_density)
            env_last_occupancy[var_env_name] = (var_occupancy, var_threshold, var_y_env)
            #
            wandb.log({
                f"test_results_per_env/{var_env_name}/accuracy": var_count_metrics["accuracy"],
                f"test_results_per_env/{var_env_name}/mae": var_count_metrics["mae"],
                f"test_results_per_env/{var_env_name}/occupancy_accuracy": var_count_metrics["occupancy_accuracy"],
                f"test_results_per_env/{var_env_name}/occupancy_f1": var_count_metrics["occupancy_f1"],
                f"test_results_per_env/{var_env_name}/threshold": var_threshold,
                f"test_results_per_env/{var_env_name}/teacher_train_time": teacher_time_1 - teacher_time_0,
                f"test_results_per_env/{var_env_name}/student_train_time": student_time_1 - student_time_0,
            }, step=var_r + 100000)
            #
            print(f"  [{var_env_name}] COUNT Acc {var_count_metrics['accuracy']:.4f} - "
                  f"MAE {var_count_metrics['mae']:.4f} - "
                  f"Occ Acc {var_count_metrics['occupancy_accuracy']:.4f} - "
                  f"Occ F1 {var_count_metrics['occupancy_f1']:.4f} (Thr {var_threshold:.2f})")
        #
        if var_r == var_repeat - 1:
            ## keep the last repeat's student so a live capture can be run through it
            save_density_checkpoint(student, var_x_shape, var_env, save_path, var_tag="student_model")
        del teacher_optimizer, student_optimizer, kd_loss
        del teacher_best_weight, student_best_weight, env_loader, var_occupancy
        del student, teacher
        torch.cuda.empty_cache()
        gc.collect()

    #
    ## -------------------------------------- Aggregate per room ----------------------------------------
    #
    var_num_classes = preset["nn"]["num_count_classes"]
    var_metric_names = ("accuracy", "mae", "occupancy_accuracy", "occupancy_f1")
    results = {}
    for var_env_name, var_rep_list in env_rep_metrics.items():
        var_env_result = {}
        for var_name in var_metric_names:
            var_arr = np.array([var_rep[var_name] for var_rep in var_rep_list])
            var_env_result[f"avg_{var_name}"] = float(var_arr.mean())
            var_std = float(var_arr.std(ddof=1)) if len(var_arr) > 1 else 0.0
            var_env_result[f"std_{var_name}"] = var_std
            var_env_result[f"se_{var_name}"] = var_std / np.sqrt(len(var_arr)) if len(var_arr) > 1 else 0.0
        var_env_result["avg_threshold"] = float(np.mean([var_rep["threshold"] for var_rep in var_rep_list]))
        var_env_result["avg_teacher_train_time"] = float(
            np.mean([var_rep["teacher_train_time"] for var_rep in var_rep_list]))
        var_env_result["avg_student_train_time"] = float(
            np.mean([var_rep["student_train_time"] for var_rep in var_rep_list]))
        var_env_result["per_class_accuracy"] = {
            var_class: float(np.mean([var_rep["per_class_accuracy"][var_class] for var_rep in var_rep_list]))
            for var_class in range(var_num_classes)
        }
        results[var_env_name] = var_env_result
        #
        var_true_density, var_pred_density = env_last_density[var_env_name]
        visualize_density_map(var_true_density, var_pred_density,
                              os.path.join(save_path, var_env_name),
                              var_threshold_frac=preset["density"]["peak_threshold"],
                              var_tag=var_env_name)
        var_occupancy_last, var_threshold_last, var_y_last = env_last_occupancy[var_env_name]
        visualize_model_performance(var_occupancy_last, var_y_last,
                                    save_dir=os.path.join(save_path, var_env_name),
                                    var_mode="occupancy", var_threshold=var_threshold_last)
        #
        print(f"\n[{var_env_name}] avg over {var_repeat} repeats: "
              f"Accuracy {var_env_result['avg_accuracy']:.4f} ± {var_env_result['se_accuracy']:.4f} | "
              f"MAE {var_env_result['avg_mae']:.4f} ± {var_env_result['se_mae']:.4f} | "
              f"Occ Acc {var_env_result['avg_occupancy_accuracy']:.4f} ± {var_env_result['se_occupancy_accuracy']:.4f} | "
              f"Occ F1 {var_env_result['avg_occupancy_f1']:.4f} ± {var_env_result['se_occupancy_f1']:.4f}")
    #
    wandb.finish()
    #
    return results
