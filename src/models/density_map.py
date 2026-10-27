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

                The objective defaults to BCE on the map logits; pass var_loss_mode="dem" (or use
                density_map_dem.py) to regress the raw map with smooth L1 instead. Everything else is
                shared between the two objectives.
"""
#
##

import copy
import csv
import gc
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import center_of_mass, label, maximum, maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import TensorDataset
from ptflops import get_model_complexity_info

from src.models.bce_that import THAT, resolve_encoder_norm
from src.models.modules.molecules import Backbone
from src.models.that_variance import build_thatv, parse_backbone
from src.models.losses.supervised_loss import OccupancyDistillationLoss
from configs.preset import preset
from src.utils import *
import wandb


torch.set_float32_matmul_precision("high")

## The shared frame stores distances in cm divided by 1000 (see preset["layouts"]), so one grid unit
## spans 10 m on both axes: a normalized coordinate v is v * 10 m from the transmitter corner. The
## WiMANS rooms are 5.10 x 10.30 m, i.e. x in [0, 0.51] and y in [0, 1.03].
METERS_PER_UNIT = 10.0

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
        var_target = torch.einsum("bl,lhw->bhw", var_occupancy, var_peak_kernels)
    else:
        var_target = torch.einsum("bl,blhw->bhw", var_occupancy, var_peak_kernels)
    ## Two occupied locations closer than the kernel width add up to > 1 in the overlap, which leaves
    ## the [0,1] range a Bernoulli target must stay in. BCEWithLogitsLoss is unbounded below for a
    ## target > 1: its gradient w.r.t. the logit is sigmoid(x) - target < 0 everywhere, so the logit
    ## grows without bound and the loss runs away to -inf. Clamp so the map stays a probability.
    return var_target.clamp_(0.0, 1.0)


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
    : var_output_mode: "probability" applies a sigmoid to the head (the BCE objective operates on the
      logits); "raw" returns the head output untouched, which is what the DEM smooth-L1 objective
      regresses directly.
    """
    #
    ##
    def __init__(self,
                 var_x_shape,
                 var_layout=None,
                 embedding_dim=100,
                 grid_size=32,
                 hidden_dim=None,
                 sigma=0.06,
                 dropout=None,
                 var_output_mode="probability"):

        super().__init__()
        if hidden_dim is None:
            hidden_dim = preset["density"].get("decoder_hidden", 64)
        if dropout is None:
            dropout = preset["density"].get("decoder_dropout", 0.1)
        #
        var_backbone = preset["nn"].get("backbone", "default")
        var_is_thatv, _ = parse_backbone(var_backbone)
        ## "cnn" is the AMAR CNN backbone (src/models/modules/molecules.py). Unlike THAT/THATV it emits
        ## a token sequence (B, N, D) rather than one vector, so forward() sums it over time to reach
        ## the (B, D) the density decoder expects. The sum (not a mean) matches THAT's aggregation and
        ## keeps the total-mass cue the count is read from.
        if var_backbone == "cnn":
            self.backbone = Backbone(input_channels=var_x_shape[-1], output_channels=embedding_dim)
            self.var_pool_backbone = True
        else:
            self.backbone = (build_thatv(var_x_shape, [embedding_dim], var_backbone)
                             if var_is_thatv
                             else THAT(var_x_shape, [embedding_dim]))
            self.var_pool_backbone = False
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.output_mode = var_output_mode
        #
        ## The layout is optional: the WiMANS paths pass one to render and read their discrete
        ## location kernels, while the H-WILD localization path predicts a free continuous map and
        ## passes none, so no kernels are needed.
        if var_layout is not None:
            var_kernels, self.location_names = build_kernels(var_layout, grid_size, sigma)
            self.register_buffer("kernels", torch.from_numpy(var_kernels))
        else:
            self.location_names = []
            self.register_buffer("kernels", None)
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
        : density: (batch, grid, grid) predicted occupancy map. Under "probability" it is the sigmoid
          of the raw map; under "raw" it is the head output itself (the DEM regression target).
        : count: (batch,) integral of the density map (soft mass, not thresholded)
        : density_map: (batch, grid, grid) raw head output; the BCE objective is applied to this, and
          the DEM objective to `density`.
        """
        var_features = self.backbone(x)
        if self.var_pool_backbone:
            ## CNN backbone: (B, N, D) token sequence -> (B, D) by summing over time
            var_features = var_features.sum(dim=1)
        var_seed = self.decoder_fc(var_features).view(-1, self.hidden_dim, 4, 4)
        var_density_map = self.decoder_conv(var_seed).squeeze(1)
        if var_density_map.shape[-1] != self.grid_size:
            var_density_map = F.interpolate(var_density_map.unsqueeze(1),
                                            size=(self.grid_size, self.grid_size),
                                            mode="bilinear", align_corners=False).squeeze(1)
        #
        if self.output_mode == "raw":
            var_density = var_density_map
        else:
            var_density = torch.sigmoid(var_density_map)
        var_count = var_density.sum(dim=(1, 2))
        #
        return var_density, var_count, var_density_map


#
## ---------------------------------------------------------------------------------------------- ##
## --------------------------------------------- metrics ---------------------------------------- ##
#
##
def _peaks_with_values(var_map, var_threshold_frac, var_abs_floor=0.0):
    """
    [description]
    : local maxima of a single density map, with the value of each one. A cell is a peak when it is the
      maximum of its 3x3 neighbourhood and clears max(threshold_frac * map_max, abs_floor). Adjacent
      peak cells form one component, which contributes its centroid as the position and its maximum as
      the value, so a broad blob is one peak, not many.
    : var_abs_floor: absolute lower bound on the map maximum and on each peak value (normally the
      validation-calibrated count threshold). 0.0 keeps the relative-only behaviour.
    : return: (positions (M, 2) normalized, values (M,))
    """
    #
    var_map = np.asarray(var_map, dtype=np.float32)
    var_max = float(var_map.max())
    if var_max <= 0 or var_max < var_abs_floor:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    #
    var_cut = max(var_threshold_frac * var_max, var_abs_floor)
    var_mask = (var_map == maximum_filter(var_map, size=3)) & (var_map > var_cut)
    var_labels, var_num = label(var_mask)
    if var_num == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    #
    var_height, var_width = var_map.shape
    var_centroids = center_of_mass(var_map, var_labels, range(1, var_num + 1))
    var_values = maximum(var_map, var_labels, range(1, var_num + 1))
    ## cell i spans [i/N, (i+1)/N) and its centre is (i+0.5)/N (build_kernels / splat_positions), so a
    ## continuous index c decodes to (c + 0.5) / N; c / N would bias every position by half a cell.
    var_positions = np.array([[(var_c[1] + 0.5) / var_width, (var_c[0] + 0.5) / var_height]
                              for var_c in var_centroids],
                             dtype=np.float32)
    #
    return var_positions, np.asarray(var_values, dtype=np.float32)


def _extract_peaks(var_map, var_threshold_frac, var_abs_floor=0.0):
    """
    [description]
    : return the (x, y) normalized coordinates of the local maxima of a single density map. See
      _peaks_with_values for the peak definition; this is the positions-only view used for plotting.
    """
    return _peaks_with_values(var_map, var_threshold_frac, var_abs_floor)[0]


def normalized_to_meters(var_xy):
    """
    [description]
    : convert one normalized (x, y) position, or an (N, 2) array of them, in the shared TX-anchored
      frame to meters. The frame is normalized by 1000 cm, so the scale is uniform (METERS_PER_UNIT
      on both axes).
    """
    return np.asarray(var_xy, dtype=np.float32) * METERS_PER_UNIT


def _panel_positions_meters(var_map, var_threshold_frac, var_layout=None, var_occupancy=None,
                            var_abs_floor=0.0):
    """
    [description]
    : the positions to circle on one density-map panel, in meters. A ground-truth panel uses the known
      layout directly when the occupancy is supplied (exact, and each position is labelled with its
      WiMANS location letter); otherwise the positions are the peaks of the map, unlabelled.
    : var_abs_floor: absolute gate forwarded to _extract_peaks on the peak panels.
    : return: (positions_m (M, 2), labels (M,) of str)
    """
    if var_layout is not None and var_occupancy is not None:
        var_names = sorted(var_layout)
        var_flags = np.asarray(var_occupancy)
        var_occupied = [var_name for var_name, var_flag in zip(var_names, var_flags) if var_flag > 0.5]
        return normalized_to_meters([var_layout[var_name] for var_name in var_occupied]), var_occupied
    var_peaks = _extract_peaks(var_map, var_threshold_frac, var_abs_floor)
    return normalized_to_meters(var_peaks), ["" for _ in range(len(var_peaks))]


def _format_positions_meters(var_positions, var_max_shown=8):
    """
    [description]
    : compact string of a list of (x, y) positions in meters, e.g. "(4.10, 6.00), (2.55, 4.55)".
      Only the first var_max_shown are listed so a noisy map cannot flood the log.
    """
    if var_positions is None or len(var_positions) == 0:
        return "[]"
    var_positions = np.asarray(var_positions)
    var_shown = ", ".join(f"({var_x:.2f}, {var_y:.2f})" for var_x, var_y in var_positions[:var_max_shown])
    if len(var_positions) > var_max_shown:
        var_shown += f", ... (+{len(var_positions) - var_max_shown} more)"
    return var_shown


def occupancy_to_positions(var_occupancy, var_layout):
    """
    [description]
    : the true (x, y) positions in meters of every occupied location of a batch, from the 0/1 occupancy
      targets and the room layout. Used only to score a layout-free prediction; the model itself never
      sees the layout.
    : var_occupancy: array (N, num_locations) of 0/1 occupancy targets
    : var_layout: dict {letter: (x_norm, y_norm)} of the room
    : return: list of (M, 2) float32 arrays in meters, one per sample (empty array for a count of 0)
    """
    var_names = sorted(var_layout)
    var_out = []
    for var_row in np.asarray(var_occupancy):
        var_occupied = [var_name for var_name, var_flag in zip(var_names, var_row) if var_flag > 0.5]
        if var_occupied:
            var_out.append(normalized_to_meters([var_layout[var_name] for var_name in var_occupied]))
        else:
            var_out.append(np.zeros((0, 2), dtype=np.float32))
    return var_out


def position_error_metrics(var_pred_positions, var_true_positions):
    """
    [description]
    : localization error of a layout-free prediction, in meters. Predicted peaks are matched to the
      true occupied positions one-to-one with a minimum-cost assignment (Hungarian); an unmatched
      position costs the frame diagonal sqrt(2) * METERS_PER_UNIT, so a missing or spurious peak is
      penalized as a full-frame miss rather than ignored.
    : var_pred_positions: list of (M, 2) predicted peak positions in meters, one per sample
    : var_true_positions: list of (K, 2) true positions in meters, one per sample
    : return: dict with the mean per-true-position error, the fraction of true positions left
      unmatched, and the fraction of predicted peaks that are spurious (over-prediction)
    """
    var_penalty = float(np.sqrt(2.0) * METERS_PER_UNIT)
    var_errors, var_misses, var_spurious, var_true_total = [], 0, 0, 0
    for var_pred, var_true in zip(var_pred_positions, var_true_positions):
        var_pred = np.asarray(var_pred, dtype=np.float64).reshape(-1, 2)
        var_true = np.asarray(var_true, dtype=np.float64).reshape(-1, 2)
        var_true_total += len(var_true)
        if len(var_true) == 0:
            ## every peak on an empty frame is spurious
            var_spurious += len(var_pred)
            continue
        if len(var_pred) == 0:
            var_errors.extend([var_penalty] * len(var_true))
            var_misses += len(var_true)
            continue
        var_size = max(len(var_true), len(var_pred))
        var_cost = np.full((var_size, var_size), var_penalty)
        var_cost[:len(var_true), :len(var_pred)] = cdist(var_true, var_pred)
        var_rows, var_cols = linear_sum_assignment(var_cost)
        for var_row, var_col in zip(var_rows, var_cols):
            if var_row >= len(var_true):
                ## an extra predicted peak matched to a dummy true position
                var_spurious += 1
                continue
            var_cost_value = float(var_cost[var_row, var_col])
            var_errors.append(var_cost_value)
            var_misses += int(var_cost_value >= var_penalty - 1e-9)
    #
    return {
        "position_error_m": float(np.mean(var_errors)) if var_errors else float("nan"),
        "position_miss_rate": float(var_misses / var_true_total) if var_true_total else float("nan"),
        "position_false_positive_rate": (float(var_spurious / var_true_total)
                                         if var_true_total else float("nan")),
    }


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
    : group-count metrics: exact-count accuracy, balanced (macro) count accuracy, count MAE,
      per-location occupancy accuracy/F1 and per-count accuracy. balanced_accuracy is the macro-average
      of the per-count accuracy over the classes present, i.e. the empty class counts as much as the
      crowded ones; it drives threshold calibration and checkpoint selection.
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
    ## macro-average over the count classes actually present: weights the rare empty class as heavily
    ## as the crowded ones, unlike the overall exact-count accuracy
    var_present = [var_class for var_class in range(var_num_classes) if np.any(var_true_count == var_class)]
    var_balanced = float(np.mean([var_per_class[var_class] for var_class in var_present])) if var_present else 0.0
    #
    return {
        "accuracy": var_exact,
        "balanced_accuracy": var_balanced,
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


def predict_counts_staged(var_occupancy, var_map_max, var_count_threshold, var_empty_threshold=None):
    """
    [description]
    : two-stage count decision. Stage 1 (empty gate): a sample whose density-map maximum is below
      var_empty_threshold is predicted empty, whatever its per-location readouts say. Stage 2: the
      remaining samples are counted by binarizing the per-location occupancy at var_count_threshold.
      The gate decouples "is the room empty" from "how many locations are occupied", which stops a
      raised map floor under domain shift from turning every empty frame into a count >= 1.
      With var_empty_threshold=None this is exactly the single-threshold rule of predict_counts.
    : var_occupancy: numpy array (N, num_locations) of per-location occupancy values
    : var_map_max: numpy array (N,) per-sample maximum of the density map
    : return: numpy array (N,) of integer predicted counts
    """
    var_counts = predict_counts(var_occupancy, var_count_threshold)
    if var_empty_threshold is not None:
        var_counts = np.where(np.asarray(var_map_max) < var_empty_threshold, 0, var_counts)
    return var_counts


def density_map_maxima(var_density):
    """
    [description]
    : per-sample maximum of an (N, H, W) density map (numpy array or torch tensor), the statistic the
      empty gate thresholds.
    : return: numpy array (N,)
    """
    if torch.is_tensor(var_density):
        return var_density.flatten(1).amax(1).cpu().numpy()
    var_density = np.asarray(var_density)
    return var_density.reshape(len(var_density), -1).max(axis=1)


def _balanced_accuracy(var_true_count, var_pred_count, var_num_classes, var_min_class=0):
    """
    [description]
    : macro-average of the per-count-class accuracy over the classes present in var_true_count at or
      above var_min_class. Unlike overall exact-count accuracy it weights the rare empty class as
      heavily as the crowded ones.
    """
    var_present = [var_class for var_class in range(var_min_class, var_num_classes)
                   if np.any(var_true_count == var_class)]
    if not var_present:
        return 0.0
    return float(np.mean([np.mean(var_pred_count[var_true_count == var_class] == var_class)
                          for var_class in var_present]))


def _finite_or_nan(var_value):
    """
    [description]
    : float(value), or nan when the value is None or not finite, so a disabled/degenerate threshold can
      still be logged and plotted.
    """
    if var_value is None:
        return float("nan")
    var_value = float(var_value)
    return var_value if np.isfinite(var_value) else float("nan")


def _ema_update(var_prev, var_value, var_decay):
    """
    [description]
    : one EMA step. Returns var_prev unchanged when var_value is missing/non-finite, so a degenerate
      per-epoch value (e.g. an empty-gate threshold of +-inf) cannot poison the whole smoothed curve.
    """
    if var_value is None or not np.isfinite(float(var_value)):
        return var_prev if var_prev is not None else float("nan")
    if var_prev is None:
        return float(var_value)
    return var_decay * float(var_value) + (1.0 - var_decay) * var_prev


def calibrate_count_threshold(var_occupancy, var_true_count,
                              var_grid=np.linspace(0.05, 0.99, 95), var_nonempty_only=True):
    """
    [description]
    : pick the occupancy threshold (stage 2) that maximizes the balanced per-count accuracy on the
      split. With var_nonempty_only the empty frames are excluded, because stage 1 already decides
      those, leaving the threshold free to separate counts 1..5. The sweep reaches 0.99 so a strict
      "only count a location when confident" rule is reachable. Ties go to the threshold closest to the
      neutral 0.5, so a small validation split cannot drag the decision boundary to an extreme.
    : return: (threshold, balanced_accuracy) at the best threshold
    """
    var_true_count = np.asarray(var_true_count).astype(int)
    var_occupancy = np.asarray(var_occupancy)
    var_num_classes = preset["nn"]["num_count_classes"]
    var_min_class = 1 if var_nonempty_only else 0
    var_sel = var_true_count >= var_min_class
    if not var_sel.any():
        return 0.5, 0.0
    var_occ_sel, var_true_sel = var_occupancy[var_sel], var_true_count[var_sel]
    var_best_threshold, var_best_score = 0.5, -1.0
    for var_threshold in var_grid:
        var_score = _balanced_accuracy(var_true_sel, predict_counts(var_occ_sel, var_threshold),
                                       var_num_classes, var_min_class)
        if (var_score > var_best_score
                or (var_score == var_best_score
                    and abs(var_threshold - 0.5) < abs(var_best_threshold - 0.5))):
            var_best_threshold, var_best_score = float(var_threshold), var_score
    return var_best_threshold, var_best_score


def calibrate_empty_threshold(var_map_max, var_true_count, var_grid=None):
    """
    [description]
    : pick the empty-gate threshold (stage 1): a sample is predicted empty when its density-map maximum
      is below the threshold. Candidate thresholds are swept over the observed maxima and the one
      maximizing the balanced accuracy of the empty / non-empty decision is kept. Calibrating the gate
      on the map maximum, rather than on the five per-location readouts, is what makes it robust to the
      readout scale drifting between rooms.
    : return: (threshold, balanced_accuracy). With no empty (or no non-empty) frames in the split the
      threshold degenerates to -inf (gate never fires) or +inf (always empty) and the score is 0.0.
    """
    var_map_max = np.asarray(var_map_max, dtype=np.float64)
    var_is_empty = np.asarray(var_true_count).astype(int) == 0
    if not var_is_empty.any():
        return -np.inf, 0.0
    if var_is_empty.all():
        return np.inf, 0.0
    if var_grid is None:
        var_lo, var_hi = float(var_map_max.min()), float(var_map_max.max())
        var_grid = np.linspace(var_lo, var_hi, 201)
    var_best_threshold, var_best_score, var_best_margin = float(var_grid[0]), -1.0, -1.0
    for var_threshold in var_grid:
        var_pred_empty = var_map_max < var_threshold
        var_score = 0.5 * (np.mean(var_pred_empty[var_is_empty])
                           + np.mean(~var_pred_empty[~var_is_empty]))
        ## among the thresholds that separate equally well, keep the one furthest from any sample, so
        ## the gate sits in the middle of the gap instead of on its edge
        var_margin = float(np.min(np.abs(var_map_max - var_threshold)))
        if (var_score > var_best_score
                or (var_score == var_best_score and var_margin > var_best_margin)):
            var_best_threshold, var_best_score, var_best_margin = (
                float(var_threshold), float(var_score), var_margin)
    return var_best_threshold, var_best_score


def resolve_count_decision(var_occupancy, var_true_count, var_map_max):
    """
    [description]
    : the two-stage count decision used for the reported metrics: (empty-gate threshold, occupancy
      threshold). preset["density"]["empty_threshold"] and preset["density"]["eval_threshold"] pin the
      two thresholds; otherwise they are calibrated on the given split with the balanced objective.
      preset["density"]["staged_count"] enables the gate: only an explicit False disables it (single
      occupancy threshold, calibrated on every frame); True, None and an absent key all keep the gate
      and calibrate it. None is the preset's documented "calibrate" value, so it must not be read as
      falsy -- doing that silently disables the gate AND sets nonempty_only=False, which is the
      single-threshold behaviour False is meant to select. Pinning matters for hyperparameter search:
      calibrating on the split that is then scored is optimistic and lets a config win by fitting the
      sweep rather than the model.
    : return: (empty_threshold or None, count_threshold)
    """
    var_true_count = np.asarray(var_true_count).astype(int)
    ## `is not False`, not truthiness: None means "calibrate the gate", False means "disable it"
    var_staged = preset["density"].get("staged_count", True) is not False
    #
    var_fixed_count = preset["density"].get("eval_threshold")
    if var_fixed_count is not None:
        var_count_threshold = float(var_fixed_count)
    else:
        var_count_threshold = calibrate_count_threshold(
            var_occupancy, var_true_count, var_nonempty_only=var_staged)[0]
    #
    var_empty_threshold = None
    if var_staged:
        var_fixed_empty = preset["density"].get("empty_threshold")
        if var_fixed_empty is not None:
            var_empty_threshold = float(var_fixed_empty)
        else:
            var_empty_threshold = calibrate_empty_threshold(var_map_max, var_true_count)[0]
    return var_empty_threshold, var_count_threshold


def layout_free_calibration(var_valid_density, var_true_count):
    """
    [description]
    : calibrate the two layout-free count rules on the validation split, from the predicted maps alone
      (no room layout involved):
      - peak_threshold: a peak counts when its value is strictly above it. Swept over the midpoints of
        the observed peak values, keeping the one that maximizes balanced count accuracy, so the
        threshold can be raised until a near-empty map yields no peaks.
      - per_person_mass: the map mass one person contributes. The value that minimizes the count MAE
        of round(map_sum / mass) on the validation split, grid-searched between the observed
        mass-per-person ratios. A plain median is biased low when occupied locations sit near the
        frame edge (their kernel is clipped), so rounding under-counts those frames.
    : var_valid_density: numpy (N, H, W) predicted maps of the validation split
    : var_true_count: (N,) integer true counts
    : return: (peak_threshold, per_person_mass)
    """
    var_true_count = np.asarray(var_true_count).astype(int)
    var_num_classes = preset["nn"]["num_count_classes"]
    var_threshold_frac = preset["density"]["peak_threshold"]
    #
    var_peak_values, var_map_sum = [], []
    for var_map in np.asarray(var_valid_density):
        var_peak_values.append(_peaks_with_values(var_map, var_threshold_frac, 0.0)[1])
        var_map_sum.append(float(np.sum(var_map)))
    var_map_sum = np.asarray(var_map_sum, dtype=np.float64)
    #
    ## candidates sit between the observed peak values, plus one below the smallest (count everything)
    ## and one above the largest (count nothing)
    var_all_values = np.concatenate(var_peak_values) if var_peak_values else np.zeros((0,), dtype=np.float32)
    if len(var_all_values):
        var_sorted = np.unique(var_all_values)
        var_candidates = np.concatenate([[var_sorted[0] - 1.0],
                                         (var_sorted[:-1] + var_sorted[1:]) / 2.0,
                                         [var_sorted[-1] + 1.0]])
    else:
        var_candidates = np.array([1e-6])
    var_best_threshold, var_best_score, var_best_margin = float(var_candidates[0]), -1.0, -1.0
    for var_threshold in var_candidates:
        var_counts = np.array([int(np.sum(var_values > var_threshold)) for var_values in var_peak_values])
        var_score = _balanced_accuracy(var_true_count, var_counts, var_num_classes, 0)
        var_margin = (float(np.min(np.abs(var_all_values - var_threshold))) if len(var_all_values)
                      else float("inf"))
        if (var_score > var_best_score
                or (var_score == var_best_score and var_margin > var_best_margin)):
            var_best_threshold, var_best_score, var_best_margin = float(var_threshold), var_score, var_margin
    #
    var_nonempty = var_true_count > 0
    if var_nonempty.any():
        ## grid-search the mass that minimizes the count MAE: a plain median of map_sum/count is
        ## biased low when the occupied locations sit near the frame edge (their kernel is clipped),
        ## so rounding under-counts those frames; the grid picks the value that balances the error
        var_ratios = var_map_sum[var_nonempty] / var_true_count[var_nonempty]
        var_lo, var_hi = float(var_ratios.min()), float(var_ratios.max())
        var_candidates = (np.linspace(var_lo * 0.5, var_hi * 1.5, 400)
                          if var_hi > var_lo else np.array([var_lo]))
        var_mass, var_best_mae = float(var_candidates[0]), np.inf
        for var_candidate in var_candidates:
            var_mae = float(np.mean(np.abs(np.round(var_map_sum / var_candidate) - var_true_count)))
            if var_mae < var_best_mae:
                var_mass, var_best_mae = float(var_candidate), var_mae
    else:
        var_mass = float(var_map_sum.max()) if len(var_map_sum) else 1.0
    var_mass = max(var_mass, 1e-6)
    #
    return var_best_threshold, var_mass


def layout_free_counts(var_density, var_peak_threshold, var_mass, var_empty_threshold=None):
    """
    [description]
    : the two layout-free counts of a batch, plus the predicted peak positions.
      - peak count: number of map peaks whose value is strictly above var_peak_threshold.
      - mass count: round(map_sum / per_person_mass).
      The positions are the SAME peaks the peak count counts (the ones above var_peak_threshold), so a
      reported count and its reported positions describe one prediction. Scoring the positions on the
      looser relative-cut peak set instead makes position_error_m non-increasing in the number of
      peaks -- an extra peak only raises position_false_positive_rate, which the .txt report does not
      print -- so a diffuse map buys a better position score for free.
      Both counts are forced to 0, and the predicted positions emptied, when the map maximum is below
      var_empty_threshold (the same map-max gate the template rule uses), so the reported count and the
      "where" stay consistent: a frame the gate declares empty has neither a count nor a position.
    : var_density: numpy (N, H, W) predicted maps
    : var_peak_threshold: absolute peak value threshold from layout_free_calibration
    : var_mass: per-person map mass from layout_free_calibration
    : var_empty_threshold: optional map-max gate; None disables it
    : return: (peak_counts (N,), mass_counts (N,), peak_positions list of (M, 2) arrays in meters)
    """
    var_threshold_frac = preset["density"]["peak_threshold"]
    var_peak_counts, var_mass_counts, var_positions = [], [], []
    for var_map in np.asarray(var_density):
        var_peaks, var_values = _peaks_with_values(var_map, var_threshold_frac, 0.0)
        ## one peak set for both: the peaks whose value clears the calibrated absolute threshold
        var_keep = var_values > var_peak_threshold
        var_peak_counts.append(int(var_keep.sum()))
        var_mass_counts.append(int(round(float(np.sum(var_map)) / var_mass)))
        var_positions.append(normalized_to_meters(var_peaks[var_keep]))
    var_peak_counts = np.asarray(var_peak_counts, dtype=int)
    var_mass_counts = np.asarray(var_mass_counts, dtype=int)
    if var_empty_threshold is not None:
        var_is_empty = density_map_maxima(var_density) < var_empty_threshold
        var_peak_counts = np.where(var_is_empty, 0, var_peak_counts)
        var_mass_counts = np.where(var_is_empty, 0, var_mass_counts)
        var_positions = [np.zeros((0, 2), dtype=np.float32) if var_empty else var_pos
                         for var_empty, var_pos in zip(var_is_empty, var_positions)]
    return var_peak_counts, var_mass_counts, var_positions


def evaluate_count_rule(var_rule, var_pred_density, var_true_y, var_layout, var_kernels,
                        var_occ_threshold, var_empty_threshold,
                        var_peak_threshold=None, var_mass=None):
    """
    [description]
    : full metric dict of one evaluated room under the chosen count rule.
      "template": count = number of the room's known locations whose per-location occupancy readout
        clears var_occ_threshold (the original rule; needs the layout).
      "layout_free": count = number of map peaks above var_peak_threshold, with the mass count and the
        position error reported alongside; the per-location occupancy readout is still reported as an
        auxiliary metric but plays no part in the count. The empty gate (var_empty_threshold) zeroes
        both counts and drops the predicted positions of a frame it declares empty.
    : var_pred_density: numpy (N, H, W) predicted maps
    : var_true_y: (N, num_locations) 0/1 occupancy targets (ground truth, for scoring only)
    : var_layout: dict of the room, used to score positions (never fed to the model)
    : var_kernels: tensor (num_locations, H, W) used for the auxiliary occupancy readout
    : return: (metrics dict, extras dict with the raw occupancy / positions / counts)
    """
    var_pred_density = np.asarray(var_pred_density, dtype=np.float32)
    var_true_y = np.asarray(var_true_y)
    var_true_count = var_true_y.sum(axis=1).round().astype(int)
    var_occupancy = sample_location_occupancy(
        torch.from_numpy(var_pred_density), torch.as_tensor(var_kernels)).numpy()
    var_map_max = density_map_maxima(var_pred_density)
    #
    if var_rule == "template":
        var_metrics = count_metrics(
            var_true_count,
            predict_counts_staged(var_occupancy, var_map_max, var_occ_threshold, var_empty_threshold),
            var_true_y, var_occupancy, var_occ_threshold)
        return var_metrics, {"occupancy": var_occupancy}
    #
    var_peak_counts, var_mass_counts, var_positions = layout_free_counts(
        var_pred_density, var_peak_threshold, var_mass, var_empty_threshold)
    var_peak_metrics = count_metrics(var_true_count, var_peak_counts, var_true_y, var_occupancy, var_occ_threshold)
    var_mass_metrics = count_metrics(var_true_count, var_mass_counts, None, None, var_occ_threshold)
    var_position_metrics = position_error_metrics(var_positions, occupancy_to_positions(var_true_y, var_layout))
    var_metrics = {
        "accuracy": var_peak_metrics["accuracy"],
        "balanced_accuracy": var_peak_metrics["balanced_accuracy"],
        "mae": var_peak_metrics["mae"],
        "per_class_accuracy": var_peak_metrics["per_class_accuracy"],
        "mass_accuracy": var_mass_metrics["accuracy"],
        "mass_balanced_accuracy": var_mass_metrics["balanced_accuracy"],
        "mass_mae": var_mass_metrics["mae"],
        "mass_per_class_accuracy": var_mass_metrics["per_class_accuracy"],
        "occupancy_accuracy": var_peak_metrics["occupancy_accuracy"],
        "occupancy_f1": var_peak_metrics["occupancy_f1"],
        **var_position_metrics,
    }
    return var_metrics, {"occupancy": var_occupancy, "positions": var_positions,
                         "peak_counts": var_peak_counts, "mass_counts": var_mass_counts}


def visualize_density_map(data_true_density,
                          data_pred_density,
                          save_dir,
                          var_num_samples=6,
                          var_threshold_frac=0.25,
                          var_tag="density_map",
                          var_layout=None,
                          var_true_occupancy=None,
                          var_peak_floor=None,
                          var_pred_peak_counts=None,
                          var_pred_mass_counts=None):
    """
    [description]
    : plot the ground-truth and predicted density maps for a spread of test samples (one row per
      sample, ground truth on the left, prediction on the right). Both panels share a metric axis in
      meters over the shared TX-anchored frame (x to the right, y away from the transmitter), every
      position is circled and annotated with its (x, y) in meters, and the ground-truth panel labels
      each occupied location with its WiMANS letter. The predicted and true positions are also printed
      per sample, so the comparison is readable without opening the figure. All panels share one
      colour scale, so a low-amplitude near-empty prediction is not stretched to full contrast.
    : var_layout: optional dict {letter: (x_norm, y_norm)} of the room, used to place the exact
      ground-truth positions in meters.
    : var_true_occupancy: optional (N, num_locations) 0/1 occupancy targets aligned with the density
      maps; with var_layout, the ground-truth panel shows the labelled true positions instead of the
      peaks of the rendered target.
    : var_peak_floor: optional absolute floor for the peak panels, normally the calibrated count
      threshold from resolve_count_decision. A map whose maximum is below it shows no peaks, and no
      cell below it counts, so the peak count uses the same bar as the reported count instead of the
      much looser empty-gate threshold.
    : var_pred_peak_counts / var_pred_mass_counts: optional (N,) inferred counts per sample; when
      given, the prediction panel title shows the layout-free peak count and mass count, so the figure
      states the number the model actually predicts rather than only how many circles are drawn.
    [return]
    : out_path: str, path of the saved figure
    """
    #
    ##
    os.makedirs(save_dir, exist_ok=True)
    var_num_samples = min(var_num_samples, len(data_true_density))
    ## the shared frame spans [0, 1] on both axes, i.e. [0, 10] m
    var_extent_m = (0.0, METERS_PER_UNIT, 0.0, METERS_PER_UNIT)
    ## one colour scale for the whole figure, so a low-amplitude prediction reads as weak
    var_vmax = max(float(np.max(data_true_density)), float(np.max(data_pred_density)), 1e-6)
    var_abs_floor = 0.0 if var_peak_floor is None else float(var_peak_floor)
    #
    ## spread the rows over the whole count range so the figure shows empty, single and crowded frames
    var_true_count = data_true_density.sum(axis=(1, 2))
    var_order = np.argsort(var_true_count)
    var_pick = var_order[np.linspace(0, len(var_order) - 1, var_num_samples).astype(int)]
    #
    var_fig, var_axes = plt.subplots(var_num_samples, 2, figsize=(9, 3.2 * var_num_samples), squeeze=False)
    var_fig.suptitle(f"Density map - ground truth vs prediction (env {var_tag}, axes in meters)")
    #
    for var_row, var_idx in enumerate(var_pick):
        var_true_occ = None if var_true_occupancy is None else np.asarray(var_true_occupancy)[var_idx]
        var_peak_count_row = (None if var_pred_peak_counts is None
                              else int(np.asarray(var_pred_peak_counts)[var_idx]))
        var_mass_count_row = (None if var_pred_mass_counts is None
                              else int(np.asarray(var_pred_mass_counts)[var_idx]))
        var_maps = ((data_true_density[var_idx], "ground truth", var_true_occ),
                    (data_pred_density[var_idx], "prediction", None))
        var_true_pos, var_pred_pos = None, None
        #
        for var_col, (var_map, var_title, var_occ) in enumerate(var_maps):
            var_ax = var_axes[var_row][var_col]
            ## origin="lower" puts y away from the transmitter upward, matching the shared frame; the
            ## extent maps the grid onto meters so the scatter below can use meter coordinates directly
            var_ax.imshow(var_map, origin="lower", cmap="viridis", vmin=0, vmax=var_vmax,
                          extent=var_extent_m, aspect="equal")
            var_ax.set_xticks(np.arange(0, METERS_PER_UNIT + 1, 2))
            var_ax.set_yticks(np.arange(0, METERS_PER_UNIT + 1, 2))
            var_ax.set_xlabel("x (m)")
            var_ax.set_ylabel("y (m)")
            #
            var_positions_m, var_labels = _panel_positions_meters(
                var_map, var_threshold_frac, var_layout, var_occ, var_abs_floor)
            if len(var_positions_m):
                var_ax.scatter(var_positions_m[:, 0], var_positions_m[:, 1],
                               s=90, facecolors="none", edgecolors="red", linewidths=1.5)
                for var_pos, var_label in zip(var_positions_m, var_labels):
                    var_prefix = f"{var_label} " if var_label else ""
                    var_ax.annotate(f"{var_prefix}({var_pos[0]:.2f}, {var_pos[1]:.2f}) m",
                                    (var_pos[0], var_pos[1]), textcoords="offset points", xytext=(6, 6),
                                    fontsize=7, color="red")
            var_ax.set_title(f"{var_title} - {len(var_positions_m)} "
                             + (("location" if len(var_positions_m) == 1 else "locations")
                                if var_occ is not None
                                else ("peak" if len(var_positions_m) == 1 else "peaks")))
            if var_title == "ground truth":
                var_true_pos = var_positions_m
            else:
                var_pred_pos = var_positions_m
                if var_peak_count_row is not None:
                    var_count_note = f"count {var_peak_count_row}"
                    if var_mass_count_row is not None:
                        var_count_note += f", mass {var_mass_count_row}"
                    var_ax.set_title(f"{var_ax.get_title()} ({var_count_note})")
        #
        print(f"  [sample {var_idx}] true {_format_positions_meters(var_true_pos)} m | "
              f"pred {_format_positions_meters(var_pred_pos)} m"
              + (f" | count peaks {var_peak_count_row} mass {var_mass_count_row}"
                 if var_peak_count_row is not None else ""))
    #
    var_fig.tight_layout(rect=[0, 0, 1, 0.96])
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


def set_count_sampling_weights(data_set):
    """
    [description]
    : per-sample sampling weight that equalizes the count classes (0..num_count_classes-1) for the
      coordinate/set head. The same measure as count_class_sampling_weights, which applies it to the
      density head: WiMANS has only 5.3% count-0 frames per room, so a plain shuffle lets the set head
      pay almost no price for never predicting an empty room. Inverse-frequency weighting fixes that
      without discarding data.
    : data_set: TensorDataset of (CSI, positions, mask) or (CSI, positions, mask, room), or a Subset /
      ConcatDataset of one. The mask is the per-sample slot-validity vector, so its sum is the count.
    : return: numpy array (N,) of non-negative sampling weights
    """
    var_count = _set_count_classes(data_set)
    var_num_classes = preset["nn"]["num_count_classes"]
    var_freq = np.bincount(var_count, minlength=var_num_classes).astype(np.float64)
    var_inverse = np.where(var_freq > 0, 1.0 / np.maximum(var_freq, 1.0), 0.0)
    #
    return var_inverse[var_count]


def _set_count_classes(data_set):
    """
    [description]
    : count class (0..5) of every sample in a TensorDataset, Subset or ConcatDataset of them carrying
      the set head's (CSI, positions, mask[, room]) tuple. Mirrors _dataset_count_classes, which reads
      the density head's occupancy vector instead.
    : return: numpy array (N,) of ints
    """
    if isinstance(data_set, torch.utils.data.ConcatDataset):
        return np.concatenate([_set_count_classes(var_part) for var_part in data_set.datasets])
    if isinstance(data_set, torch.utils.data.Subset):
        return _set_count_classes(data_set.dataset)[data_set.indices]
    return data_set.tensors[2].sum(axis=1).round().long().numpy()


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
        "output_mode": model.output_mode,
        "location_names": list(model.location_names),
        ## The backbone and the encoder CNN norm are read from the preset by DensityMapNet (and
        ## make_channel_norm), so they are not recoverable from the weights alone. Record them or a
        ## reload cannot rebuild the module a checkpoint trained with nn.backbone=thatv* or a
        ## non-batch nn.encoder_norm describes (load_density_checkpoint still infers them for
        ## checkpoints written before this key existed).
        "backbone": preset["nn"].get("backbone", "default"),
        "encoder_norm": resolve_encoder_norm(),
        "embedding_dim": model.decoder_fc[0].in_features,
    }, var_out_path)
    print(f"  Checkpoint saved to {var_out_path}")
    return var_out_path


def infer_backbone_from_state_dict(var_state_dict):
    """
    [description]
    : recover the nn.backbone string a checkpoint was trained with from its parameter names. The
      THATV features leave unambiguous names, so the ablation string is exact; a state dict with THAT
      encoders but none of them is "default", and one with no THAT encoders is the AMAR CNN ("cnn").
      Needed for checkpoints written before save_density_checkpoint recorded "backbone".
    : return: str, a value DensityMapNet accepts as nn.backbone
    """
    var_keys = set(var_state_dict)
    var_use_stats = "backbone.layer_stats.0.weight" in var_keys
    var_use_tokens = "backbone.layer_left_cnn_0.layer_conv.weight" in var_keys
    var_use_pool = "backbone.layer_pool_proj.weight" in var_keys
    if not (var_use_stats or var_use_tokens or var_use_pool):
        return ("default" if "backbone.layer_left_encoder.0.layer_norm_0.weight" in var_keys
                else "cnn")
    var_disabled = [var_name for var_name, var_on in
                    (("stats", var_use_stats), ("tokens", var_use_tokens), ("pool", var_use_pool))
                    if not var_on]
    return "thatv" if not var_disabled else "thatv-" + ",".join("no" + var_name for var_name in var_disabled)


def infer_encoder_norm_from_state_dict(var_state_dict):
    """
    [description]
    : "batch" when the encoder CNN branches carry BatchNorm running statistics, else a sample-wise
      norm. GroupNorm and LayerNorm share parameter names and shapes, so either rebuilds the weights
      identically; "group" is returned as the representative. Needed for checkpoints written before
      save_density_checkpoint recorded "encoder_norm".
    : return: str
    """
    return ("batch"
            if "backbone.layer_left_encoder.0.layer_cnn.0.1.running_mean" in set(var_state_dict)
            else "group")


def load_density_checkpoint(var_checkpoint_path, var_device, var_backbone=None, var_encoder_norm=None):
    """
    [description]
    : rebuild the DensityMapNet a checkpoint describes and load its weights. backbone and encoder norm
      come from the checkpoint when present, else from the state-dict keys (see the infer_* helpers),
      else the current preset; an explicit var_backbone / var_encoder_norm wins over all three.
    : return: (model in eval mode, checkpoint dict)
    """
    var_checkpoint = torch.load(var_checkpoint_path, map_location=var_device)
    var_state_dict = var_checkpoint["model_state_dict"]
    var_backbone = (var_backbone or var_checkpoint.get("backbone")
                    or infer_backbone_from_state_dict(var_state_dict))
    var_encoder_norm = (var_encoder_norm or var_checkpoint.get("encoder_norm")
                        or infer_encoder_norm_from_state_dict(var_state_dict))
    ## DensityMapNet reads the backbone from the preset and make_channel_norm reads the encoder norm
    ## from the preset or the AMAR_ENCODER_NORM env, so set both before constructing; the env var is
    ## set too so a stale export cannot rebuild a different norm than the checkpoint was trained with.
    preset["nn"]["backbone"] = var_backbone
    preset["nn"]["encoder_norm"] = var_encoder_norm
    os.environ["AMAR_ENCODER_NORM"] = var_encoder_norm
    var_layout_name = var_checkpoint["layout_name"]
    var_layout = (preset["layouts"].get(var_layout_name)
                  or preset["layouts"][preset["data"]["environment"][0]])
    var_model = DensityMapNet(tuple(var_checkpoint["x_shape"]), var_layout,
                              embedding_dim=var_checkpoint.get("embedding_dim", 100),
                              grid_size=var_checkpoint["grid_size"],
                              hidden_dim=var_checkpoint.get("decoder_hidden"),
                              sigma=var_checkpoint["sigma"],
                              dropout=var_checkpoint.get("decoder_dropout"),
                              var_output_mode=var_checkpoint.get("output_mode", "probability")).to(var_device)
    var_model.load_state_dict(var_state_dict)
    var_model.eval()
    return var_model, var_checkpoint


def plot_training_curves(var_history, var_save_path, var_tag="training"):
    """
    [description]
    : write the per-epoch training history to a PNG and a CSV. Called after every epoch by
      train_density, so both files on disk always show the run so far and can be opened while the
      training is still going, without needing a display on the training node.
    : var_history: dict of metric name -> list, one entry per epoch (see train_density).
    : var_save_path: directory of the run; nothing is written when it is None or empty.
    : var_tag: filename prefix, e.g. "train_r0" or "student_empty_room_r0".
    : return: (png path, csv path), or None when there is nothing to plot.
    """
    if not var_save_path or not var_history.get("epoch"):
        return None
    os.makedirs(str(var_save_path), exist_ok=True)
    #
    ## (panel title, y label, [(history key, legend label, line style)])
    var_panels = [
        ("train loss (epoch mean)", "loss", [("train_loss", "train_loss", "-")]),
        ("valid loss (fixed batch)", "loss", [("valid_loss", "valid_loss", "-")]),
        ("distillation term", "KD",
         [("train_kd", "train_kd", "-"), ("valid_kd", "valid_kd", "--")]),
        ("valid exact-count accuracy", "accuracy",
         [("valid_accuracy", "raw", "-"), ("valid_accuracy_smoothed", "smoothed (EMA)", "--")]),
        ("valid balanced accuracy (selection)", "balanced acc",
         [("valid_balanced_accuracy", "raw", "-"),
          ("valid_balanced_accuracy_smoothed", "smoothed (EMA)", "--")]),
        ("valid count MAE", "MAE",
         [("valid_mae", "raw", "-"), ("valid_mae_smoothed", "smoothed (EMA)", "--")]),
        ("valid occupancy accuracy", "accuracy",
         [("valid_occupancy_accuracy", "raw", "-"),
          ("valid_occupancy_accuracy_smoothed", "smoothed (EMA)", "--")]),
        ("valid occupancy F1", "F1",
         [("valid_occupancy_f1", "raw", "-"), ("valid_occupancy_f1_smoothed", "smoothed (EMA)", "--")]),
        ("occupancy threshold", "threshold",
         [("valid_occupancy_threshold", "raw", "-"),
          ("valid_occupancy_threshold_smoothed", "smoothed (EMA)", "--")]),
        ("empty-gate threshold", "map max threshold",
         [("valid_empty_threshold", "raw", "-"),
          ("valid_empty_threshold_smoothed", "smoothed (EMA)", "--")]),
        ("peak threshold (layout-free)", "peak value",
         [("valid_peak_threshold", "raw", "-"),
          ("valid_peak_threshold_smoothed", "smoothed (EMA)", "--")]),
        ("learning rate", "lr", [("learning_rate", "lr", "-")]),
    ]
    var_epochs = var_history["epoch"]
    var_fig, var_axes = plt.subplots(3, 4, figsize=(20, 12), squeeze=False)
    var_flat_axes = var_axes.ravel()
    for var_ax, (var_title, var_ylabel, var_series) in zip(var_flat_axes, var_panels):
        for var_key, var_label, var_style in var_series:
            var_ax.plot(var_epochs, var_history[var_key], var_style, label=var_label)
        var_ax.set_title(var_title)
        var_ax.set_xlabel("epoch")
        var_ax.set_ylabel(var_ylabel)
        var_ax.grid(alpha=0.3)
        var_ax.legend(fontsize=8)
    for var_ax in var_flat_axes[len(var_panels):]:
        var_ax.axis("off")
    var_fig.suptitle(f"{var_tag} - metrics evolution")
    var_fig.tight_layout(rect=[0, 0, 1, 0.95])
    var_png = os.path.join(str(var_save_path), f"{var_tag}_curves.png")
    var_fig.savefig(var_png, dpi=110)
    plt.close(var_fig)
    #
    ## the CSV mirrors the figure, one row per epoch, so the numbers survive the run
    var_csv = os.path.join(str(var_save_path), f"{var_tag}_curves.csv")
    var_keys = ["epoch"] + [var_key for _, _, var_series in var_panels for var_key, _, _ in var_series]
    with open(var_csv, "w", newline="") as var_file:
        var_writer = csv.writer(var_file)
        var_writer.writerow(var_keys)
        for var_row in range(len(var_epochs)):
            var_writer.writerow([var_history[var_key][var_row] for var_key in var_keys])
    #
    return var_png, var_csv


def augment_csi(var_x):
    """
    [description]
    : on-the-fly CSI augmentation shared by the WiMANS density trainer and the H-WILD localization
      trainer. The encoder's LayerNorm removes a constant input scale, so the previous +0.1 noise was
      negligible against amplitudes around 60 and the gain was largely cancelled. Scale the noise to
      the signal and add a small temporal shift; keep the gain and channel dropout.
    : var_x: tensor (batch, time, features)
    : return: tensor, same shape
    """
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
                  var_kernel_bank=None,
                  var_save_dir=None,
                  var_tag="training",
                  var_loss_mode="bce"):
    """
    [description]
    : train the density-map model. The objective is the binary cross-entropy between the predicted
      density map and the Gaussian target rendered from the occupancy labels ("bce"), or, under
      var_loss_mode="dem", the smooth-L1 (Huber) direct error between the raw predicted map and the
      same target; the count and the per-location occupancy are read back off the map either way.
      The occupancy threshold is calibrated on the
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
    : var_save_dir: optional directory where the per-epoch curves (PNG + CSV) are rewritten after
      every epoch; skipped when None, so the history is still kept in memory for the return value.
    : var_tag: filename prefix of the curve files, e.g. "train_r0".
    : var_loss_mode: "bce" (default) applies BCEWithLogits to the raw head output; "dem" regresses the
      raw density map with smooth L1. Under "dem" the model must be built with var_output_mode="raw",
      and the KD term (if any) is a smooth-L1 match of the two density maps instead of the Bernoulli
      distillation of OccupancyDistillationLoss.
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
    ## the count rule the checkpoint is selected under, kept in sync with the rule the runners report
    var_count_rule = preset["density"].get("count_rule", "layout_free")
    #
    var_best_score = -np.inf
    var_best_mae = np.inf
    var_best_weight = None
    var_epoch_saved = 0
    var_counter = 0
    ## EMA-smoothed series (0.3 => ~3-epoch memory): the raw per-epoch values are noisy on a small
    ## validation split, so every logged curve also gets a smoothed counterpart
    var_ema_accuracy = None
    var_ema_mae = None
    var_ema_balanced = None
    var_ema_occ_acc = None
    var_ema_occ_f1 = None
    var_ema_threshold = None
    var_ema_empty_threshold = None
    var_ema_peak_threshold = None
    var_ema_decay = 0.3
    #
    ## per-epoch history, rewritten to var_save_dir after every epoch so the run can be watched live
    var_history = {var_key: [] for var_key in (
        "epoch", "train_loss", "valid_loss", "train_kd", "valid_kd",
        "valid_accuracy", "valid_accuracy_smoothed",
        "valid_balanced_accuracy", "valid_balanced_accuracy_smoothed",
        "valid_mae", "valid_mae_smoothed",
        "valid_occupancy_accuracy", "valid_occupancy_accuracy_smoothed",
        "valid_occupancy_f1", "valid_occupancy_f1_smoothed",
        "valid_occupancy_threshold", "valid_occupancy_threshold_smoothed",
        "valid_empty_threshold", "valid_empty_threshold_smoothed",
        "valid_peak_threshold", "valid_peak_threshold_smoothed", "learning_rate")}
    #
    var_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=preset["nn"]["scheduler"]["num_warmup_epochs"] * len(var_train_loader),
        num_training_steps=var_epochs * len(var_train_loader),
        min_lr_ratio=preset["nn"]["scheduler"]["min_lr_ratio"],
    )

    for var_epoch in range(var_epochs):
        var_time_e0 = time.time()
        model.train()
        ## accumulate the epoch-mean losses: the last batch alone is far too noisy to read
        var_train_loss_sum = 0.0
        var_train_kd_sum = 0.0
        var_train_batches = 0
        for var_batch in var_train_loader:
            var_x, var_occupancy = var_batch[0], var_batch[1]
            var_x = augment_csi(var_x.to(device))
            var_occupancy = var_occupancy.to(device)
            #
            ## each sample is rendered with its own room's kernels when the batch mixes rooms
            if len(var_batch) > 2:
                var_kernels = var_kernel_bank[var_batch[2].to(var_kernel_bank.device)]
            else:
                var_kernels = model.kernels
            #
            var_density, _, var_density_map = model(var_x)
            var_density_target = render_density_targets(var_occupancy, var_kernels)
            if var_loss_mode == "dem":
                ## direct error minimization: regress the raw map with a robust (Huber) loss
                var_loss = F.smooth_l1_loss(var_density, var_density_target)
            else:
                var_loss = F.binary_cross_entropy_with_logits(var_density_map, var_density_target)
            #
            ## few-shot distillation: the frozen teacher supervises the student's density map
            var_kd_value = 0.0
            if var_use_distillation:
                with torch.no_grad():
                    var_teacher_density, _, var_teacher_map = teacher(var_x)
                if var_loss_mode == "dem":
                    var_kd = F.smooth_l1_loss(var_density, var_teacher_density)
                else:
                    var_kd = kd_loss(var_density_map, var_teacher_map)
                var_kd_value = float(var_kd.detach())
                var_loss = var_loss + kd_weight * var_kd
            var_train_loss_sum += float(var_loss.detach())
            var_train_kd_sum += var_kd_value
            var_train_batches += 1
            #
            optimizer.zero_grad()
            var_loss.backward()
            optimizer.step()
            var_scheduler.step()
        var_train_loss_mean = var_train_loss_sum / max(1, var_train_batches)
        var_train_kd_mean = var_train_kd_sum / max(1, var_train_batches)
        #
        model.eval()
        with torch.no_grad():
            var_valid_batch = next(iter(var_valid_loader))
            var_valid_x, var_valid_y = var_valid_batch[0], var_valid_batch[1]
            var_valid_density, _, var_valid_map = model(var_valid_x.to(device))
            var_valid_kd_value = 0.0
            if var_use_distillation:
                var_valid_teacher_density, _, var_valid_teacher_map = teacher(var_valid_x.to(device))
                if var_loss_mode == "dem":
                    var_valid_kd_value = float(
                        F.smooth_l1_loss(var_valid_density, var_valid_teacher_density).detach())
                else:
                    var_valid_kd_value = float(kd_loss(var_valid_map, var_valid_teacher_map).detach())
            if len(var_valid_batch) > 2:
                var_valid_kernels = var_kernel_bank[var_valid_batch[2].to(var_kernel_bank.device)]
            else:
                var_valid_kernels = model.kernels
            var_valid_occupancy = sample_location_occupancy(
                var_valid_density, var_valid_kernels).cpu().numpy()
            var_valid_map_max = density_map_maxima(var_valid_density)
            ## validation loss on the same fixed batch, so a clean decreasing curve exists
            var_valid_target = render_density_targets(var_valid_y.to(device), var_valid_kernels)
            if var_loss_mode == "dem":
                var_valid_loss = float(F.smooth_l1_loss(var_valid_density, var_valid_target).detach())
            else:
                var_valid_loss = float(
                    F.binary_cross_entropy_with_logits(var_valid_map, var_valid_target).detach())
            var_valid_y = var_valid_y.numpy()
        #
        var_true_count = var_valid_y.sum(axis=1).round()
        var_empty_threshold, var_threshold = resolve_count_decision(
            var_valid_occupancy, var_true_count, var_valid_map_max)
        ## Count rule. Under "layout_free" the count is the number of predicted locations: the map
        ## peaks above a validation-calibrated absolute threshold, so the checkpoint is selected on
        ## the same locate-then-count rule the report scores. Under "template" it stays the
        ## per-location occupancy readout (the earlier protocol).
        if var_count_rule == "layout_free":
            var_valid_density_np = var_valid_density.cpu().numpy()
            var_peak_threshold, var_mass = layout_free_calibration(
                var_valid_density_np, var_true_count)
            var_pred_count = layout_free_counts(
                var_valid_density_np, var_peak_threshold, var_mass, var_empty_threshold)[0]
        else:
            var_peak_threshold = None
            var_pred_count = predict_counts_staged(
                var_valid_occupancy, var_valid_map_max, var_threshold, var_empty_threshold)
        var_metrics = count_metrics(
            var_true_count, var_pred_count, var_valid_y, var_valid_occupancy, var_threshold)
        #
        ## smooth every logged series before comparing epochs; the checkpoint is selected on the
        ## smoothed balanced accuracy so the rare empty class counts as much as the crowded ones
        var_empty_threshold_log = _finite_or_nan(var_empty_threshold)
        var_ema_accuracy = _ema_update(var_ema_accuracy, var_metrics["accuracy"], var_ema_decay)
        var_ema_balanced = _ema_update(var_ema_balanced, var_metrics["balanced_accuracy"], var_ema_decay)
        var_ema_mae = _ema_update(var_ema_mae, var_metrics["mae"], var_ema_decay)
        var_ema_occ_acc = _ema_update(var_ema_occ_acc, var_metrics["occupancy_accuracy"], var_ema_decay)
        var_ema_occ_f1 = _ema_update(var_ema_occ_f1, var_metrics["occupancy_f1"], var_ema_decay)
        var_ema_threshold = _ema_update(var_ema_threshold, var_threshold, var_ema_decay)
        var_ema_empty_threshold = _ema_update(var_ema_empty_threshold, var_empty_threshold_log, var_ema_decay)
        var_peak_threshold_log = _finite_or_nan(var_peak_threshold)
        var_ema_peak_threshold = _ema_update(var_ema_peak_threshold, var_peak_threshold_log, var_ema_decay)
        #
        wandb.log({
            "epoch": var_epoch,
            "train_loss": var_train_loss_mean,
            "valid_loss": var_valid_loss,
            "train_kd": var_train_kd_mean,
            "valid_kd": var_valid_kd_value,
            "valid_accuracy": var_metrics["accuracy"],
            "valid_balanced_accuracy": var_metrics["balanced_accuracy"],
            "valid_mae": var_metrics["mae"],
            "valid_occupancy_accuracy": var_metrics["occupancy_accuracy"],
            "valid_occupancy_f1": var_metrics["occupancy_f1"],
            "valid_occupancy_threshold": var_threshold,
            "valid_empty_threshold": var_empty_threshold_log,
            "valid_peak_threshold": var_peak_threshold_log,
            "valid_accuracy_smoothed": var_ema_accuracy,
            "valid_balanced_accuracy_smoothed": var_ema_balanced,
            "valid_mae_smoothed": var_ema_mae,
            "valid_occupancy_accuracy_smoothed": var_ema_occ_acc,
            "valid_occupancy_f1_smoothed": var_ema_occ_f1,
            "valid_occupancy_threshold_smoothed": var_ema_threshold,
            "valid_empty_threshold_smoothed": var_ema_empty_threshold,
            "valid_peak_threshold_smoothed": var_ema_peak_threshold,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        #
        print(f"Epoch {var_epoch}/{var_epochs} - %.3fs" % (time.time() - var_time_e0),
              "- Loss %.6f" % var_train_loss_mean,
              "- ValidLoss %.6f" % var_valid_loss,
              "- Valid Acc %.4f" % var_metrics["accuracy"],
              "- Valid Bal %.4f" % var_metrics["balanced_accuracy"],
              "- Valid MAE %.4f" % var_metrics["mae"],
              "- Thr %.2f" % var_threshold,
              "- EmptyThr %.3f" % var_empty_threshold_log,
              "- PeakThr %.4f" % var_peak_threshold_log)
        #
        ## append the epoch to the history and rewrite the curves, so the run can be watched live
        var_history["epoch"].append(var_epoch)
        var_history["train_loss"].append(var_train_loss_mean)
        var_history["valid_loss"].append(var_valid_loss)
        var_history["train_kd"].append(var_train_kd_mean)
        var_history["valid_kd"].append(var_valid_kd_value)
        var_history["valid_accuracy"].append(var_metrics["accuracy"])
        var_history["valid_accuracy_smoothed"].append(var_ema_accuracy)
        var_history["valid_balanced_accuracy"].append(var_metrics["balanced_accuracy"])
        var_history["valid_balanced_accuracy_smoothed"].append(var_ema_balanced)
        var_history["valid_mae"].append(var_metrics["mae"])
        var_history["valid_mae_smoothed"].append(var_ema_mae)
        var_history["valid_occupancy_accuracy"].append(var_metrics["occupancy_accuracy"])
        var_history["valid_occupancy_accuracy_smoothed"].append(var_ema_occ_acc)
        var_history["valid_occupancy_f1"].append(var_metrics["occupancy_f1"])
        var_history["valid_occupancy_f1_smoothed"].append(var_ema_occ_f1)
        var_history["valid_occupancy_threshold"].append(var_threshold)
        var_history["valid_occupancy_threshold_smoothed"].append(var_ema_threshold)
        var_history["valid_empty_threshold"].append(var_empty_threshold_log)
        var_history["valid_empty_threshold_smoothed"].append(var_ema_empty_threshold)
        var_history["valid_peak_threshold"].append(var_peak_threshold_log)
        var_history["valid_peak_threshold_smoothed"].append(var_ema_peak_threshold)
        var_history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        plot_training_curves(var_history, var_save_dir, var_tag)
        #
        if (var_ema_balanced > var_best_score
                or (var_ema_balanced == var_best_score and var_ema_mae < var_best_mae)):
            var_best_score = var_ema_balanced
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
    print(f"Best smoothed balanced count accuracy: {var_best_score:.6f}, smoothed count MAE: {var_best_mae:.6f}")
    #
    return var_best_weight


#
## ---------------------------------------------------------------------------------------------- ##
## ---------------------------------------- main runner ----------------------------------------- ##
#
##
def build_optimizer(var_parameters, var_preset):
    """
    [description]
    : build the training optimizer for the density-map model from preset["nn"]["optimizer"], so Adam
      and SGD can be compared from the config alone (e.g. --hp nn.optimizer=sgd). Both share the
      preset learning rate and weight decay; SGD additionally honours preset["nn"]["momentum"]
      (default 0.0, i.e. plain SGD).
    : var_parameters: iterable of parameters handed to the optimizer
    : var_preset: the preset dict
    : return: torch.optim.Optimizer
    """
    var_nn = var_preset["nn"]
    var_name = str(var_nn.get("optimizer", "adam")).lower()
    if var_name == "adam":
        return torch.optim.Adam(var_parameters,
                                lr=var_nn["lr"],
                                weight_decay=var_nn["weight_decay"])
    if var_name == "sgd":
        return torch.optim.SGD(var_parameters,
                               lr=var_nn["lr"],
                               weight_decay=var_nn["weight_decay"],
                               momentum=var_nn.get("momentum", 0.0))
    raise ValueError(f"unknown nn.optimizer {var_name!r}; expected 'adam' or 'sgd'")


def run_density_map(data_train_x,
                    data_train_y,
                    data_test_x,
                    data_test_y,
                    var_repeat=10, var_task="count", var_env="empty_room",
                    save_path="./visualizations/temp",
                    var_loss_mode="bce"):
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
    : var_loss_mode: "bce" (default) or "dem" (smooth-L1 direct error minimization)
    : return: dict, averaged count and per-location occupancy metrics with SE
    """
    #
    ## coordinate-regression head: one head that directly predicts user locations and counts them
    if preset["density"].get("head", "density") == "coords":
        var_res = run_density_map_coords({var_env: (data_train_x, data_train_y)},
                                         {var_env: (data_test_x, data_test_y)},
                                         var_repeat=var_repeat, save_path=save_path, var_env=var_env)
        return var_res[var_env]
    #
    ##
    var_output_mode = "raw" if var_loss_mode == "dem" else "probability"
    var_run_prefix = "DensityMapDEM" if var_loss_mode == "dem" else "DensityMap"
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
    var_count_rule = preset["density"].get("count_rule", "layout_free")

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
    result_per_class = []
    var_rep_metrics = []
    var_valid_rep_metrics = []
    #
    var_macs, var_params = get_model_complexity_info(
        DensityMapNet(var_x_shape, var_layout, grid_size=var_grid_size,
                      var_output_mode=var_output_mode), var_x_shape, as_strings=False)
    print("Parameters:", var_params, "- FLOPs:", var_macs * 2)

    for var_r in range(var_repeat):
        #
        ##
        print("Repeat", var_r)
        name_run = f"{var_run_prefix}{var_r}_" + "_".join(preset["data"]["environment"])
        wandb.init(project="density_map", name=name_run, config=preset, reinit=True)
        #
        torch.random.manual_seed(var_r + 39)
        #
        model_density = DensityMapNet(var_x_shape, var_layout,
                                      embedding_dim=100, grid_size=var_grid_size,
                                      var_output_mode=var_output_mode).to(device)
        optimizer = build_optimizer(model_density.parameters(), preset)
        #
        var_time_0 = time.time()
        var_best_weight = train_density(model=model_density,
                                        optimizer=optimizer,
                                        data_train_set=data_train_set,
                                        data_valid_set=data_valid_set,
                                        var_batch_size=preset["nn"]["batch_size"],
                                        var_epochs=preset["nn"]["epoch"],
                                        device=device,
                                        var_save_dir=save_path, var_tag=f"train_r{var_r}",
                                        var_loss_mode=var_loss_mode)
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
        pred_density = pred_density.numpy()
        var_time_2 = time.time()
        #
        ## Calibrate the two-stage count decision on the validation split with the selected checkpoint,
        ## so the test count uses the same rule that selected the epoch.
        with torch.no_grad():
            var_valid_density, _, _ = model_density(torch.from_numpy(data_valid_x).to(device))
            var_valid_occupancy = sample_location_occupancy(
                var_valid_density, model_density.kernels).cpu().numpy()
        var_empty_threshold, var_threshold = resolve_count_decision(
            var_valid_occupancy, data_valid_y.sum(axis=1).round(), density_map_maxima(var_valid_density))
        ## layout-free count rule: calibrate the peak threshold and the per-person mass on the same
        ## validation split, from the predicted maps alone
        if var_count_rule == "layout_free":
            var_peak_threshold, var_mass = layout_free_calibration(
                var_valid_density.cpu().numpy(), data_valid_y.sum(axis=1).round())
        else:
            var_peak_threshold, var_mass = None, None
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
        var_count_metrics, var_extra = evaluate_count_rule(
            var_count_rule, pred_density, data_test_y, var_layout, model_density.kernels.cpu(),
            var_threshold, var_empty_threshold, var_peak_threshold, var_mass)
        var_occupancy = var_extra["occupancy"]
        #
        ## the identical decision scored on the validation split, so a pinned threshold can be chosen
        ## on held-out data instead of on the split it is then reported on (scripts/sweep_density.sh).
        ## Reported as valid_avg_* / se_valid_* alongside the test metrics.
        var_valid_metrics, _ = evaluate_count_rule(
            var_count_rule, var_valid_density.cpu().numpy(), data_valid_y, var_layout,
            model_density.kernels.cpu(), var_threshold, var_empty_threshold,
            var_peak_threshold, var_mass)
        #
        var_empty_threshold_log = _finite_or_nan(var_empty_threshold)
        var_metric_logs = {}
        if "mass_accuracy" in var_count_metrics:
            var_metric_logs = {
                "mass_accuracy": var_count_metrics["mass_accuracy"],
                "mass_balanced_accuracy": var_count_metrics["mass_balanced_accuracy"],
                "mass_mae": var_count_metrics["mass_mae"],
                "position_error_m": var_count_metrics["position_error_m"],
                "position_miss_rate": var_count_metrics["position_miss_rate"],
                "position_false_positive_rate": var_count_metrics["position_false_positive_rate"],
                "peak_threshold": var_peak_threshold,
                "per_person_mass": var_mass,
            }
        wandb.log({
            "repeat": var_r,
            "train_time": var_time_1 - var_time_0,
            "test_time": var_time_2 - var_time_1,
            "accuracy": var_count_metrics["accuracy"],
            "balanced_accuracy": var_count_metrics["balanced_accuracy"],
            "mae": var_count_metrics["mae"],
            "occupancy_accuracy": var_count_metrics["occupancy_accuracy"],
            "occupancy_f1": var_count_metrics["occupancy_f1"],
            "occupancy_threshold": var_threshold,
            "empty_threshold": var_empty_threshold_log,
            **var_metric_logs,
        }, step=var_r + 100000)
        #
        print("  COUNT: Acc %.4f - Bal %.4f - MAE %.4f - Occupancy Acc %.4f - Occupancy F1 %.4f - "
              "Thr %.2f - EmptyThr %.3f"
              % (var_count_metrics["accuracy"], var_count_metrics["balanced_accuracy"],
                 var_count_metrics["mae"], var_count_metrics["occupancy_accuracy"],
                 var_count_metrics["occupancy_f1"], var_threshold, var_empty_threshold_log))
        if "mass_accuracy" in var_count_metrics:
            print("  LAYOUT-FREE: peaks Acc %.4f | mass Acc %.4f - mass MAE %.4f | "
                  "position err %.3f m (miss %.3f, fp %.3f) | peakThr %.4f - mass/person %.3f"
                  % (var_count_metrics["accuracy"], var_count_metrics["mass_accuracy"],
                     var_count_metrics["mass_mae"], var_count_metrics["position_error_m"],
                     var_count_metrics["position_miss_rate"],
                     var_count_metrics["position_false_positive_rate"],
                     var_peak_threshold, var_mass))
        #
        if var_r == var_repeat - 1:
            var_peak_floor = var_peak_threshold if var_count_rule == "layout_free" else var_threshold
            var_fig_path = visualize_density_map(true_density, pred_density, save_path,
                                                 var_threshold_frac=preset["density"]["peak_threshold"],
                                                 var_tag=var_env,
                                                 var_layout=var_layout, var_true_occupancy=data_test_y,
                                                 var_peak_floor=var_peak_floor,
                                                 var_pred_peak_counts=var_extra.get("peak_counts"),
                                                 var_pred_mass_counts=var_extra.get("mass_counts"))
            ## the standard per-location performance figures, on the binarized occupancy
            visualize_model_performance(var_occupancy, data_test_y, save_dir=save_path,
                                        var_mode="occupancy", var_threshold=var_threshold)
            ## keep the last repeat's weights so a live capture can be run through the model
            save_density_checkpoint(model_density, var_x_shape, var_env, save_path)
            print(f"  Figure saved to {var_fig_path}")
        #
        ## carry the decision the metrics were produced under, so the report can print it: all four
        ## values are derived (calibrated or pinned), so a metric is only reproducible with them
        var_rep_metrics.append({**var_count_metrics,
                                "threshold": float(var_threshold),
                                "empty_threshold": var_empty_threshold_log,
                                "peak_threshold": _finite_or_nan(var_peak_threshold),
                                "per_person_mass": _finite_or_nan(var_mass)})
        var_valid_rep_metrics.append(var_valid_metrics)
        result_per_class.append(var_count_metrics["per_class_accuracy"])
        #
        if var_r != var_repeat - 1:
            del model_density, optimizer, test_loader, pred_density, var_occupancy, true_density
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

    ## every numeric metric of the chosen rule, so layout-free adds its mass and position metrics
    var_metric_names = ["accuracy", "balanced_accuracy", "mae",
                        "mass_accuracy", "mass_balanced_accuracy", "mass_mae",
                        "position_error_m", "position_miss_rate", "position_false_positive_rate",
                        "occupancy_accuracy", "occupancy_f1"]
    results = {}
    for var_name in var_metric_names:
        if all(var_name in var_rep for var_rep in var_rep_metrics):
            var_mean, var_se = mean_se([var_rep[var_name] for var_rep in var_rep_metrics])
            results[f"avg_{var_name}"] = var_mean
            results[f"se_{var_name}"] = var_se
        ## the same aggregation on the validation split, under the valid_ prefix
        if all(var_name in var_rep for var_rep in var_valid_rep_metrics):
            var_mean, var_se = mean_se([var_rep[var_name] for var_rep in var_valid_rep_metrics])
            results[f"valid_avg_{var_name}"] = var_mean
            results[f"se_valid_{var_name}"] = var_se
    ## the decision every metric above was produced under (mean over repeats)
    for var_key in ("threshold", "empty_threshold", "peak_threshold", "per_person_mass"):
        var_values = np.array([var_rep.get(var_key, np.nan) for var_rep in var_rep_metrics], dtype=float)
        results[f"avg_{var_key}"] = (float(np.nanmean(var_values))
                                     if np.isfinite(var_values).any() else float("nan"))
    results["per_class_accuracy"] = {
        var_class: float(np.mean([var_rep[var_class] for var_rep in result_per_class]))
        for var_class in range(var_num_classes)
    }
    if all("mass_per_class_accuracy" in var_rep for var_rep in var_rep_metrics):
        results["mass_per_class_accuracy"] = {
            var_class: float(np.mean([var_rep["mass_per_class_accuracy"][var_class]
                                      for var_rep in var_rep_metrics]))
            for var_class in range(var_num_classes)
        }
    #
    wandb.log({f"avg_{var_name}": results[f"avg_{var_name}"] for var_name in
               ("accuracy", "balanced_accuracy", "mae", "occupancy_accuracy", "occupancy_f1")})
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
                                 save_path="./visualizations/temp",
                                 var_loss_mode="bce"):
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
    : var_loss_mode: "bce" (default) or "dem" (smooth-L1 direct error minimization)
    : return: dict, env name -> averaged count and per-location occupancy metrics with SE
    """
    #
    ## coordinate-regression head: one head that directly predicts user locations and counts them
    if preset["density"].get("head", "density") == "coords":
        return run_density_map_coords(train_sets_by_env, test_sets_by_env,
                                      var_repeat=var_repeat, save_path=save_path)
    #
    var_output_mode = "raw" if var_loss_mode == "dem" else "probability"
    var_run_prefix = "DensityMapCDDEM" if var_loss_mode == "dem" else "DensityMapCD"
    device = select_device()
    print(f"Using device: {device}")
    #
    for var_env_name in list(test_sets_by_env):
        var_x_env, var_y_env = test_sets_by_env[var_env_name]
        test_sets_by_env[var_env_name] = (var_x_env, np.asarray(var_y_env, dtype=np.float32))
    #
    var_train_envs = list(train_sets_by_env)
    var_layout = preset["layouts"][var_train_envs[0]]
    var_count_rule = preset["density"].get("count_rule", "layout_free")
    var_grid_size = preset["density"]["grid_size"]
    var_sigma = preset["density"]["sigma"]
    #
    ## one unit-mass kernel bank per training room; each sample is rendered and read out with its own
    ## room's kernels, which is what lets several rooms share one model
    var_kernel_bank = torch.stack([
        torch.from_numpy(build_kernels(preset["layouts"][var_env_name], var_grid_size, var_sigma)[0])
        for var_env_name in var_train_envs]).to(device)
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
        DensityMapNet(var_x_shape, var_layout, grid_size=var_grid_size,
                      var_output_mode=var_output_mode), var_x_shape, as_strings=False)
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
        name_run = f"{var_run_prefix}{var_r}_" + "_".join(var_train_envs)
        wandb.init(project="density_map_cross_domain", name=name_run, config=preset, reinit=True)
        #
        torch.random.manual_seed(var_r + 39)
        #
        model_density = DensityMapNet(var_x_shape, var_layout,
                                      embedding_dim=100, grid_size=var_grid_size,
                                      var_output_mode=var_output_mode).to(device)
        optimizer = build_optimizer(model_density.parameters(), preset)
        #
        var_best_weight = train_density(model=model_density,
                                        optimizer=optimizer,
                                        data_train_set=data_train_set,
                                        data_valid_set=data_valid_set,
                                        var_batch_size=preset["nn"]["batch_size"],
                                        var_epochs=preset["nn"]["epoch"],
                                        device=device,
                                        var_kernel_bank=var_kernel_bank,
                                        var_save_dir=save_path, var_tag=f"train_r{var_r}",
                                        var_loss_mode=var_loss_mode)
        model_density.load_state_dict(var_best_weight)
        model_density.eval()
        #
        ## two-stage count decision from the training rooms' validation split
        with torch.no_grad():
            var_valid_density, _, _ = model_density(torch.from_numpy(var_valid_x).to(device))
            var_valid_occupancy = sample_location_occupancy(
                var_valid_density,
                var_kernel_bank[torch.from_numpy(var_valid_room).to(var_kernel_bank.device)]).cpu().numpy()
        var_empty_threshold, var_threshold = resolve_count_decision(
            var_valid_occupancy, var_valid_y.sum(axis=1).round(), density_map_maxima(var_valid_density))
        ## layout-free count rule: calibrate the peak threshold and the per-person mass on the training
        ## rooms' validation split, from the predicted maps alone
        if var_count_rule == "layout_free":
            var_peak_threshold, var_mass = layout_free_calibration(
                var_valid_density.cpu().numpy(), var_valid_y.sum(axis=1).round())
        else:
            var_peak_threshold, var_mass = None, None
        var_empty_threshold_log = _finite_or_nan(var_empty_threshold)
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
            true_density = torch.einsum("bl,lhw->bhw",
                                        torch.from_numpy(var_y_env),
                                        peak_normalized_kernels(var_test_kernels)).numpy()
            pred_density = pred_density.numpy()
            #
            var_count_metrics, var_extra = evaluate_count_rule(
                var_count_rule, pred_density, var_y_env, preset["layouts"][var_env_name],
                var_test_kernels, var_threshold, var_empty_threshold,
                var_peak_threshold, var_mass)
            var_occupancy = var_extra["occupancy"]
            #
            var_rep = {**var_count_metrics, "threshold": var_threshold,
                       "empty_threshold": var_empty_threshold_log,
                       "peak_threshold": _finite_or_nan(var_peak_threshold),
                       "per_person_mass": _finite_or_nan(var_mass)}
            env_rep_metrics.setdefault(var_env_name, []).append(var_rep)
            env_last_density[var_env_name] = (true_density, pred_density)
            env_last_occupancy[var_env_name] = (var_occupancy, var_threshold, var_y_env, var_extra)
            #
            var_metric_logs = {}
            if "mass_accuracy" in var_count_metrics:
                var_metric_logs = {
                    f"test_results_per_env/{var_env_name}/{var_name}": var_count_metrics[var_name]
                    for var_name in ("mass_accuracy", "mass_balanced_accuracy", "mass_mae",
                                     "position_error_m", "position_miss_rate",
                                     "position_false_positive_rate")}
            wandb.log({
                f"test_results_per_env/{var_env_name}/accuracy": var_count_metrics["accuracy"],
                f"test_results_per_env/{var_env_name}/balanced_accuracy": var_count_metrics["balanced_accuracy"],
                f"test_results_per_env/{var_env_name}/mae": var_count_metrics["mae"],
                f"test_results_per_env/{var_env_name}/occupancy_accuracy": var_count_metrics["occupancy_accuracy"],
                f"test_results_per_env/{var_env_name}/occupancy_f1": var_count_metrics["occupancy_f1"],
                f"test_results_per_env/{var_env_name}/empty_threshold": var_empty_threshold_log,
                **var_metric_logs,
            }, step=var_r + 100000)
            #
            print(f"  [{var_env_name}] COUNT Acc {var_count_metrics['accuracy']:.4f} - "
                  f"Bal {var_count_metrics['balanced_accuracy']:.4f} - "
                  f"MAE {var_count_metrics['mae']:.4f} - "
                  f"Occ Acc {var_count_metrics['occupancy_accuracy']:.4f} - "
                  f"Occ F1 {var_count_metrics['occupancy_f1']:.4f} "
                  f"(Thr {var_threshold:.2f}, EmptyThr {var_empty_threshold_log:.3f})")
            if "mass_accuracy" in var_count_metrics:
                print(f"  [{var_env_name}] LAYOUT-FREE mass Acc {var_count_metrics['mass_accuracy']:.4f} - "
                      f"mass MAE {var_count_metrics['mass_mae']:.4f} - "
                      f"position err {var_count_metrics['position_error_m']:.3f} m "
                      f"(miss {var_count_metrics['position_miss_rate']:.3f}, "
                      f"fp {var_count_metrics['position_false_positive_rate']:.3f})")
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
    var_metric_names = ("accuracy", "balanced_accuracy", "mae",
                        "mass_accuracy", "mass_balanced_accuracy", "mass_mae",
                        "position_error_m", "position_miss_rate", "position_false_positive_rate",
                        "occupancy_accuracy", "occupancy_f1")
    results = {}
    for var_env_name, var_rep_list in env_rep_metrics.items():
        var_env_result = {}
        for var_name in var_metric_names:
            if not all(var_name in var_rep for var_rep in var_rep_list):
                continue
            var_arr = np.array([var_rep[var_name] for var_rep in var_rep_list])
            var_env_result[f"avg_{var_name}"] = float(var_arr.mean())
            var_std = float(var_arr.std(ddof=1)) if len(var_arr) > 1 else 0.0
            var_env_result[f"std_{var_name}"] = var_std
            var_env_result[f"se_{var_name}"] = var_std / np.sqrt(len(var_arr)) if len(var_arr) > 1 else 0.0
        var_env_result["avg_threshold"] = float(np.mean([var_rep["threshold"] for var_rep in var_rep_list]))
        var_empty_values = np.array([var_rep["empty_threshold"] for var_rep in var_rep_list])
        var_env_result["avg_empty_threshold"] = (float(np.nanmean(var_empty_values))
                                                 if np.isfinite(var_empty_values).any() else float("nan"))
        ## the layout-free peak rule is derived too: record it next to the metrics it produced
        for var_key in ("peak_threshold", "per_person_mass"):
            var_values = np.array([var_rep[var_key] for var_rep in var_rep_list], dtype=float)
            var_env_result[f"avg_{var_key}"] = (float(np.nanmean(var_values))
                                                if np.isfinite(var_values).any() else float("nan"))
        var_env_result["per_class_accuracy"] = {
            var_class: float(np.mean([var_rep["per_class_accuracy"][var_class] for var_rep in var_rep_list]))
            for var_class in range(var_num_classes)
        }
        if all("mass_per_class_accuracy" in var_rep for var_rep in var_rep_list):
            var_env_result["mass_per_class_accuracy"] = {
                var_class: float(np.mean([var_rep["mass_per_class_accuracy"][var_class]
                                          for var_rep in var_rep_list]))
                for var_class in range(var_num_classes)
            }
        results[var_env_name] = var_env_result
        #
        var_true_density, var_pred_density = env_last_density[var_env_name]
        var_occupancy_last, var_threshold_last, var_y_last, var_extra_last = env_last_occupancy[var_env_name]
        var_peak_floor_last = var_peak_threshold if var_count_rule == "layout_free" else var_threshold_last
        visualize_density_map(var_true_density, var_pred_density,
                              os.path.join(save_path, var_env_name),
                              var_threshold_frac=preset["density"]["peak_threshold"],
                              var_tag=var_env_name,
                              var_layout=preset["layouts"][var_env_name],
                              var_true_occupancy=var_y_last,
                              var_peak_floor=var_peak_floor_last,
                              var_pred_peak_counts=var_extra_last.get("peak_counts"),
                              var_pred_mass_counts=var_extra_last.get("mass_counts"))
        ## the standard per-location performance figures, on the binarized occupancy of the last repeat
        visualize_model_performance(var_occupancy_last, var_y_last,
                                    save_dir=os.path.join(save_path, var_env_name),
                                    var_mode="occupancy", var_threshold=var_threshold_last)
        #
        print(f"\n[{var_env_name}] avg over {var_repeat} repeats: "
              f"Accuracy {var_env_result['avg_accuracy']:.4f} ± {var_env_result['se_accuracy']:.4f} | "
              f"Balanced {var_env_result['avg_balanced_accuracy']:.4f} ± {var_env_result['se_balanced_accuracy']:.4f} | "
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
                             save_path="./visualizations/temp",
                             var_loss_mode="bce"):
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
      Under var_loss_mode="dem" both models use the raw head and the smooth-L1 objective, and the
      distillation term is a smooth-L1 match of the two density maps.
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
    : var_loss_mode: "bce" (default) or "dem" (smooth-L1 direct error minimization)
    : return: dict, per-environment averaged count and per-location occupancy metrics with SE
    """
    #
    var_output_mode = "raw" if var_loss_mode == "dem" else "probability"
    var_run_prefix = "DensityMapFewShotDEM" if var_loss_mode == "dem" else "DensityMapFewShot"
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
    var_count_rule = preset["density"].get("count_rule", "layout_free")
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
        DensityMapNet(var_x_shape, var_layout, grid_size=var_grid_size,
                      var_output_mode=var_output_mode), var_x_shape, as_strings=False)
    print("Parameters:", var_params, "- FLOPs:", var_macs * 2)

    #
    ## ========================================= Train & Evaluate =========================================
    #
    env_rep_metrics = {}
    env_last_density = {}
    env_last_occupancy = {}

    for var_r in range(var_repeat):
        print("Repeat", var_r)
        name_run = f"{var_run_prefix}{var_r}_{env_name}_k{var_few_shot_ratio}"
        wandb.init(project="density_map_few_shot", name=name_run, config=preset, reinit=True)
        #
        torch.random.manual_seed(var_r + 39)
        #
        ## ---------------------------------------- Teacher ----------------------------------------
        #
        teacher = DensityMapNet(var_x_shape, var_layout,
                                embedding_dim=100, grid_size=var_grid_size,
                                var_output_mode=var_output_mode).to(device)
        if var_compile:
            teacher.backbone = torch.compile(teacher.backbone)
        teacher_optimizer = build_optimizer(teacher.parameters(), preset)
        teacher_time_0 = time.time()
        teacher_best_weight = train_density(model=teacher,
                                            optimizer=teacher_optimizer,
                                            data_train_set=teacher_train_set,
                                            data_valid_set=teacher_valid_set,
                                            var_batch_size=preset["nn"]["batch_size"],
                                            var_epochs=var_teacher_epochs,
                                            device=device,
                                            var_save_dir=save_path, var_tag=f"teacher_r{var_r}",
                                            var_loss_mode=var_loss_mode)
        teacher_time_1 = time.time()
        teacher.load_state_dict(teacher_best_weight)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        #
        ## ---------------------------------------- Student ----------------------------------------
        #
        student = DensityMapNet(var_x_shape, var_layout,
                                embedding_dim=100, grid_size=var_grid_size,
                                var_output_mode=var_output_mode).to(device)
        if var_compile:
            student.backbone = torch.compile(student.backbone)
        student_optimizer = build_optimizer(student.parameters(), preset)
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
                                            kd_weight=var_kd_weight,
                                            var_save_dir=save_path, var_tag=f"student_r{var_r}",
                                            var_loss_mode=var_loss_mode)
        student_time_1 = time.time()
        student.load_state_dict(student_best_weight)
        student.eval()

        #
        ## two-stage count decision from the training room's validation split, reused for every test
        # room (no test-room labels are used anywhere)
        #
        with torch.no_grad():
            var_valid_density, _, _ = student(torch.from_numpy(data_teacher_valid_x).to(device))
            var_valid_occupancy = sample_location_occupancy(
                var_valid_density, student.kernels).cpu().numpy()
        var_empty_threshold, var_threshold = resolve_count_decision(
            var_valid_occupancy, data_teacher_valid_y.sum(axis=1).round(),
            density_map_maxima(var_valid_density))
        ## layout-free count rule: calibrate the peak threshold and the per-person mass on the student's
        ## validation split, from the predicted maps alone
        if var_count_rule == "layout_free":
            var_peak_threshold, var_mass = layout_free_calibration(
                var_valid_density.cpu().numpy(), data_teacher_valid_y.sum(axis=1).round())
        else:
            var_peak_threshold, var_mass = None, None
        var_empty_threshold_log = _finite_or_nan(var_empty_threshold)

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
            true_density = torch.einsum("bl,lhw->bhw",
                                        torch.from_numpy(var_y_env),
                                        peak_normalized_kernels(var_test_kernels)).numpy()
            pred_density = pred_density.numpy()
            #
            var_count_metrics, var_extra = evaluate_count_rule(
                var_count_rule, pred_density, var_y_env, preset["layouts"][var_env_name],
                var_test_kernels, var_threshold, var_empty_threshold,
                var_peak_threshold, var_mass)
            var_occupancy = var_extra["occupancy"]
            #
            var_rep = {**var_count_metrics, "threshold": var_threshold,
                       "empty_threshold": var_empty_threshold_log,
                       "teacher_train_time": teacher_time_1 - teacher_time_0,
                       "student_train_time": student_time_1 - student_time_0,
                       "peak_threshold": _finite_or_nan(var_peak_threshold),
                       "per_person_mass": _finite_or_nan(var_mass)}
            env_rep_metrics.setdefault(var_env_name, []).append(var_rep)
            env_last_density[var_env_name] = (true_density, pred_density)
            env_last_occupancy[var_env_name] = (var_occupancy, var_threshold, var_y_env, var_extra)
            #
            var_metric_logs = {}
            if "mass_accuracy" in var_count_metrics:
                var_metric_logs = {
                    f"test_results_per_env/{var_env_name}/{var_name}": var_count_metrics[var_name]
                    for var_name in ("mass_accuracy", "mass_balanced_accuracy", "mass_mae",
                                     "position_error_m", "position_miss_rate",
                                     "position_false_positive_rate")}
            wandb.log({
                f"test_results_per_env/{var_env_name}/accuracy": var_count_metrics["accuracy"],
                f"test_results_per_env/{var_env_name}/balanced_accuracy": var_count_metrics["balanced_accuracy"],
                f"test_results_per_env/{var_env_name}/mae": var_count_metrics["mae"],
                f"test_results_per_env/{var_env_name}/occupancy_accuracy": var_count_metrics["occupancy_accuracy"],
                f"test_results_per_env/{var_env_name}/occupancy_f1": var_count_metrics["occupancy_f1"],
                f"test_results_per_env/{var_env_name}/threshold": var_threshold,
                f"test_results_per_env/{var_env_name}/empty_threshold": var_empty_threshold_log,
                f"test_results_per_env/{var_env_name}/teacher_train_time": teacher_time_1 - teacher_time_0,
                f"test_results_per_env/{var_env_name}/student_train_time": student_time_1 - student_time_0,
                **var_metric_logs,
            }, step=var_r + 100000)
            #
            print(f"  [{var_env_name}] COUNT Acc {var_count_metrics['accuracy']:.4f} - "
                  f"Bal {var_count_metrics['balanced_accuracy']:.4f} - "
                  f"MAE {var_count_metrics['mae']:.4f} - "
                  f"Occ Acc {var_count_metrics['occupancy_accuracy']:.4f} - "
                  f"Occ F1 {var_count_metrics['occupancy_f1']:.4f} "
                  f"(Thr {var_threshold:.2f}, EmptyThr {var_empty_threshold_log:.3f})")
            if "mass_accuracy" in var_count_metrics:
                print(f"  [{var_env_name}] LAYOUT-FREE mass Acc {var_count_metrics['mass_accuracy']:.4f} - "
                      f"mass MAE {var_count_metrics['mass_mae']:.4f} - "
                      f"position err {var_count_metrics['position_error_m']:.3f} m "
                      f"(miss {var_count_metrics['position_miss_rate']:.3f}, "
                      f"fp {var_count_metrics['position_false_positive_rate']:.3f})")
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
    var_metric_names = ("accuracy", "balanced_accuracy", "mae",
                        "mass_accuracy", "mass_balanced_accuracy", "mass_mae",
                        "position_error_m", "position_miss_rate", "position_false_positive_rate",
                        "occupancy_accuracy", "occupancy_f1")
    results = {}
    for var_env_name, var_rep_list in env_rep_metrics.items():
        var_env_result = {}
        for var_name in var_metric_names:
            if not all(var_name in var_rep for var_rep in var_rep_list):
                continue
            var_arr = np.array([var_rep[var_name] for var_rep in var_rep_list])
            var_env_result[f"avg_{var_name}"] = float(var_arr.mean())
            var_std = float(var_arr.std(ddof=1)) if len(var_arr) > 1 else 0.0
            var_env_result[f"std_{var_name}"] = var_std
            var_env_result[f"se_{var_name}"] = var_std / np.sqrt(len(var_arr)) if len(var_arr) > 1 else 0.0
        var_env_result["avg_threshold"] = float(np.mean([var_rep["threshold"] for var_rep in var_rep_list]))
        var_empty_values = np.array([var_rep["empty_threshold"] for var_rep in var_rep_list])
        var_env_result["avg_empty_threshold"] = (float(np.nanmean(var_empty_values))
                                                 if np.isfinite(var_empty_values).any() else float("nan"))
        ## the layout-free peak rule is derived too: record it next to the metrics it produced
        for var_key in ("peak_threshold", "per_person_mass"):
            var_values = np.array([var_rep[var_key] for var_rep in var_rep_list], dtype=float)
            var_env_result[f"avg_{var_key}"] = (float(np.nanmean(var_values))
                                                if np.isfinite(var_values).any() else float("nan"))
        var_env_result["avg_teacher_train_time"] = float(
            np.mean([var_rep["teacher_train_time"] for var_rep in var_rep_list]))
        var_env_result["avg_student_train_time"] = float(
            np.mean([var_rep["student_train_time"] for var_rep in var_rep_list]))
        var_env_result["per_class_accuracy"] = {
            var_class: float(np.mean([var_rep["per_class_accuracy"][var_class] for var_rep in var_rep_list]))
            for var_class in range(var_num_classes)
        }
        if all("mass_per_class_accuracy" in var_rep for var_rep in var_rep_list):
            var_env_result["mass_per_class_accuracy"] = {
                var_class: float(np.mean([var_rep["mass_per_class_accuracy"][var_class]
                                          for var_rep in var_rep_list]))
                for var_class in range(var_num_classes)
            }
        results[var_env_name] = var_env_result
        #
        var_true_density, var_pred_density = env_last_density[var_env_name]
        var_occupancy_last, var_threshold_last, var_y_last, var_extra_last = env_last_occupancy[var_env_name]
        var_peak_floor_last = var_peak_threshold if var_count_rule == "layout_free" else var_threshold_last
        visualize_density_map(var_true_density, var_pred_density,
                              os.path.join(save_path, var_env_name),
                              var_threshold_frac=preset["density"]["peak_threshold"],
                              var_tag=var_env_name,
                              var_layout=preset["layouts"][var_env_name],
                              var_true_occupancy=var_y_last,
                              var_peak_floor=var_peak_floor_last,
                              var_pred_peak_counts=var_extra_last.get("peak_counts"),
                              var_pred_mass_counts=var_extra_last.get("mass_counts"))
        visualize_model_performance(var_occupancy_last, var_y_last,
                                    save_dir=os.path.join(save_path, var_env_name),
                                    var_mode="occupancy", var_threshold=var_threshold_last)
        #
        print(f"\n[{var_env_name}] avg over {var_repeat} repeats: "
              f"Accuracy {var_env_result['avg_accuracy']:.4f} ± {var_env_result['se_accuracy']:.4f} | "
              f"Balanced {var_env_result['avg_balanced_accuracy']:.4f} ± {var_env_result['se_balanced_accuracy']:.4f} | "
              f"MAE {var_env_result['avg_mae']:.4f} ± {var_env_result['se_mae']:.4f} | "
              f"Occ Acc {var_env_result['avg_occupancy_accuracy']:.4f} ± {var_env_result['se_occupancy_accuracy']:.4f} | "
              f"Occ F1 {var_env_result['avg_occupancy_f1']:.4f} ± {var_env_result['se_occupancy_f1']:.4f}")
    #
    wandb.finish()
    #
    return results


#
## ---------------------------------------------------------------------------------------------- ##
## --------------------------- coordinate-regression head (one head) ---------------------------- ##
#
## The alternative to the density grid: a single head that directly regresses the user locations.
## The head emits `num_queries` coordinate hypotheses and one objectness logit each; a hypothesis is
## a predicted user when its objectness clears the validation-calibrated threshold, so the count is
## exactly the number of predicted locations -- no grid and no post-hoc peak extraction. The
## architecture (SetLocalizer) and the Hungarian set loss are shared with
## src/models/set_localization.py. Select it with preset["density"]["head"] = "coords".
#
##

def occupancy_to_position_batch(var_occupancy, var_layout):
    """
    [description]
    : turn a batch of per-location occupancy vectors into the fixed-size position set the coordinate
      head regresses: (positions (N, K, 2) normalized to the shared frame, mask (N, K)), K the largest
      count in the batch, unused slots masked out.
    : var_occupancy: (N, num_locations) array of 0/1 entries
    : var_layout: dict {letter: (x_norm, y_norm)}
    : return: (pos (N, K, 2) float32, mask (N, K) float32)
    """
    #
    var_names = sorted(var_layout)
    var_occ = np.asarray(var_occupancy)
    var_k = max(1, int(round(float(var_occ.sum(axis=1).max()))))
    var_pos = np.zeros((len(var_occ), var_k, 2), dtype=np.float32)
    var_mask = np.zeros((len(var_occ), var_k), dtype=np.float32)
    for var_i, var_row in enumerate(var_occ):
        var_pts = [var_layout[var_name] for var_name, var_flag in zip(var_names, var_row) if var_flag > 0.5]
        for var_j, var_pt in enumerate(var_pts[:var_k]):
            var_pos[var_i, var_j] = var_pt
            var_mask[var_i, var_j] = 1.0
    return var_pos, var_mask


def calibrate_coord_threshold(var_obj, var_true_count, var_grid=np.linspace(0.05, 0.95, 19),
                              var_nonempty_only=False):
    """
    [description]
    : pick the objectness threshold (stage 2) whose selected-query count maximizes the balanced
      per-count accuracy on the split, so the count decision is calibrated rather than fixed. With
      var_nonempty_only the empty frames are excluded, because stage 1 already decides those, leaving
      the threshold free to separate counts 1..num_queries. Ties go to the threshold closest to the
      neutral 0.5, so a small validation split cannot drag the decision boundary to an extreme.
    : var_obj: (N, num_queries) objectness logits
    : return: (threshold, balanced_accuracy)
    """
    #
    var_prob = 1.0 / (1.0 + np.exp(-np.asarray(var_obj, dtype=np.float64)))
    var_true_count = np.asarray(var_true_count).astype(int)
    var_num_classes = preset["nn"]["num_count_classes"]
    var_min_class = 1 if var_nonempty_only else 0
    var_sel = var_true_count >= var_min_class
    if not var_sel.any():
        return 0.5, 0.0
    var_prob_sel, var_true_sel = var_prob[var_sel], var_true_count[var_sel]
    var_best_threshold, var_best_score = float(var_grid[0]), -1.0
    for var_threshold in var_grid:
        var_counts = (var_prob_sel > var_threshold).sum(axis=1)
        var_score = _balanced_accuracy(var_true_sel, var_counts, var_num_classes, var_min_class)
        if (var_score > var_best_score
                or (var_score == var_best_score
                    and abs(var_threshold - 0.5) < abs(var_best_threshold - 0.5))):
            var_best_threshold, var_best_score = float(var_threshold), var_score
    return var_best_threshold, var_best_score


def coord_objectness_maxima(var_obj):
    """
    [description]
    : per-sample maximum objectness probability, the statistic the coordinate head's empty gate fires
      on. The analogue of density_map_maxima for the density head's map maximum.
    : var_obj: (N, num_queries) objectness logits
    : return: numpy array (N,)
    """
    var_prob = 1.0 / (1.0 + np.exp(-np.asarray(var_obj, dtype=np.float64)))
    return var_prob.max(axis=1)


def calibrate_coord_empty_threshold(var_obj, var_true_count, var_grid=None):
    """
    [description]
    : pick the empty-gate threshold (stage 1) of the coordinate head: a sample is predicted empty when
      the maximum objectness probability over its queries is below the threshold. Candidate thresholds
      are swept over the observed maxima and the one maximizing the balanced accuracy of the empty /
      non-empty decision is kept. Mirrors calibrate_empty_threshold, which does this on the density-map
      maximum; calibrating on a single pooled statistic, rather than on the per-query readouts, is what
      makes the gate robust to the objectness scale drifting between rooms.
    : return: (threshold, balanced_accuracy). With no empty (or no non-empty) frames in the split the
      threshold degenerates to -inf (gate never fires) or +inf (always empty) and the score is 0.0.
    """
    #
    var_obj_max = coord_objectness_maxima(var_obj)
    var_is_empty = np.asarray(var_true_count).astype(int) == 0
    if not var_is_empty.any():
        return -np.inf, 0.0
    if var_is_empty.all():
        return np.inf, 0.0
    if var_grid is None:
        var_lo, var_hi = float(var_obj_max.min()), float(var_obj_max.max())
        var_grid = np.linspace(var_lo, var_hi, 201)
    var_best_threshold, var_best_score, var_best_margin = float(var_grid[0]), -1.0, -1.0
    for var_threshold in var_grid:
        var_pred_empty = var_obj_max < var_threshold
        var_score = 0.5 * (np.mean(var_pred_empty[var_is_empty])
                           + np.mean(~var_pred_empty[~var_is_empty]))
        ## among the thresholds that separate equally well, keep the one furthest from any sample, so
        ## the gate sits in the middle of the gap instead of on its edge
        var_margin = float(np.min(np.abs(var_obj_max - var_threshold)))
        if (var_score > var_best_score
                or (var_score == var_best_score and var_margin > var_best_margin)):
            var_best_threshold, var_best_score, var_best_margin = (
                float(var_threshold), float(var_score), var_margin)
    return var_best_threshold, var_best_score


def resolve_coord_count_decision(var_obj, var_true_count):
    """
    [description]
    : the two-stage count decision of the coordinate head: (empty-gate threshold or None, objectness
      threshold). preset["density"]["eval_threshold"] and ["empty_threshold"] pin the two thresholds;
      otherwise they are calibrated on the given split with the balanced objective.
      preset["density"]["staged_count"] disables the gate and falls back to the single objectness
      threshold calibrated on every frame. Pinning matters for hyperparameter search: calibrating on
      the split that is then scored is optimistic and lets a config win by fitting the sweep rather
      than the model. Mirrors resolve_count_decision, so the coordinate head follows the same counting
      protocol as the density head.
    : var_obj: (N, num_queries) objectness logits
    : return: (empty_threshold or None, objectness_threshold)
    """
    #
    var_true_count = np.asarray(var_true_count).astype(int)
    var_staged = preset["density"].get("staged_count", True)
    #
    var_fixed_count = preset["density"].get("eval_threshold")
    if var_fixed_count is not None:
        var_count_threshold = float(var_fixed_count)
    else:
        var_count_threshold = calibrate_coord_threshold(
            var_obj, var_true_count, var_nonempty_only=var_staged)[0]
    #
    var_empty_threshold = None
    if var_staged:
        var_fixed_empty = preset["density"].get("empty_threshold")
        if var_fixed_empty is not None:
            var_empty_threshold = float(var_fixed_empty)
        else:
            var_empty_threshold = calibrate_coord_empty_threshold(var_obj, var_true_count)[0]
    return var_empty_threshold, var_count_threshold


def coord_metrics(var_xy, var_obj, var_true_pos, var_true_mask, var_threshold,
                  var_empty_threshold=None):
    """
    [description]
    : count and location metrics of a coordinate-head batch. A query is a predicted user when its
      objectness probability clears var_threshold; the count is the number of such queries and the
      predicted locations are their (x, y) in meters. Stage 1 (empty gate): a sample whose maximum
      objectness probability is below var_empty_threshold is predicted empty whatever its queries say,
      which decouples "is the room empty" from "how many people are in it". The position error matches
      predicted to true locations with a Hungarian assignment (position_error_metrics), penalizing
      misses and spurious locations, so the localization error keeps the multi-user structure.
    : var_xy: (N, num_queries, 2) normalized predicted positions
    : var_obj: (N, num_queries) objectness logits
    : var_true_pos: (N, K, 2) normalized true positions, var_true_mask: (N, K) 0/1
    : var_empty_threshold: optional stage-1 gate; None disables it
    : return: metrics dict
    """
    #
    var_xy = np.asarray(var_xy, dtype=np.float64)
    var_prob = 1.0 / (1.0 + np.exp(-np.asarray(var_obj, dtype=np.float64)))
    var_true_pos = np.asarray(var_true_pos, dtype=np.float64)
    var_true_mask = np.asarray(var_true_mask)
    var_true_count = var_true_mask.sum(axis=1).round().astype(int)
    var_obj_max = var_prob.max(axis=1) if var_prob.size else np.zeros(len(var_prob))
    var_pred_positions, var_true_positions, var_pred_counts = [], [], []
    for var_i in range(len(var_xy)):
        var_sel = var_prob[var_i] > var_threshold
        if var_empty_threshold is not None and var_obj_max[var_i] < var_empty_threshold:
            var_sel = np.zeros_like(var_sel)
        var_pred_positions.append(normalized_to_meters(var_xy[var_i][var_sel]))
        var_pred_counts.append(int(var_sel.sum()))
        var_true_positions.append(normalized_to_meters(var_true_pos[var_i][var_true_mask[var_i] > 0.5]))
    var_metrics = count_metrics(var_true_count, np.asarray(var_pred_counts, dtype=int))
    var_metrics.update(position_error_metrics(var_pred_positions, var_true_positions))
    return var_metrics


def splat_positions(var_positions, var_grid_size, var_sigma):
    """
    [description]
    : rasterize a set of normalized (x, y) positions into a (grid, grid) map by summing a
      peak-normalized Gaussian at each point and clamping to [0, 1] -- the same convention
      render_density_targets uses for the density head. This is the map analogue of the coordinate
      head's point set, so the figure is comparable to visualize_density_map's panels.
    : var_positions: (K, 2) array of normalized positions in the shared [0, 1]^2 frame
    : return: (grid, grid) float32 array in [0, 1]
    """
    var_positions = np.asarray(var_positions, dtype=np.float32).reshape(-1, 2)
    var_axis = (np.arange(var_grid_size) + 0.5) / var_grid_size
    var_yy, var_xx = np.meshgrid(var_axis, var_axis, indexing="ij")  # rows -> y, cols -> x
    var_map = np.zeros((var_grid_size, var_grid_size), dtype=np.float32)
    for var_cx, var_cy in var_positions:
        var_blob = np.exp(-((var_xx - var_cx) ** 2 + (var_yy - var_cy) ** 2) / (2 * var_sigma ** 2))
        var_map = var_map + var_blob / max(float(var_blob.max()), 1e-12)
    return np.clip(var_map, 0.0, 1.0)


def visualize_coord_heatmap(var_true_sets, var_pred_sets, save_dir, var_tag="coords",
                            var_grid_size=None, var_sigma=None, var_num_samples=6):
    """
    [description]
    : density-map-style heatmap of the coordinate head's output. The head has no native field, so each
      panel is a synthetic map: the predicted query set (queries whose objectness clears the calibrated
      threshold) and the ground-truth position set are rasterized with splat_positions onto the same
      grid the density head uses, with the actual points circled and annotated in meters. Rows are test
      samples spread over the true count range, ground truth on the left, prediction on the right, one
      shared [0, 1] colour scale.
    : This is a rendering of the point output, not the head's own quantity: the count in each title is
      the number of points, not a map sum, so it is the coordinate head's query count and is not the
      same decision rule as the density head's peak/mass count.
    : var_true_sets / var_pred_sets: lists of (K, 2) normalized position arrays, one per sample
    : var_tag: filename suffix, normally the room name
    : return: str, path of the saved figure
    """
    os.makedirs(save_dir, exist_ok=True)
    var_grid_size = preset["density"]["grid_size"] if var_grid_size is None else var_grid_size
    var_sigma = preset["density"]["sigma"] if var_sigma is None else var_sigma
    var_num = min(var_num_samples, len(var_true_sets))
    var_extent_m = (0.0, METERS_PER_UNIT, 0.0, METERS_PER_UNIT)
    #
    ## spread the rows over the whole count range, as visualize_density_map does
    var_true_count = np.array([len(var_set) for var_set in var_true_sets])
    var_order = np.argsort(var_true_count)
    var_pick = var_order[np.linspace(0, len(var_order) - 1, var_num).astype(int)]
    #
    var_fig, var_axes = plt.subplots(var_num, 2, figsize=(9, 3.2 * var_num), squeeze=False)
    var_fig.suptitle(f"Coordinate head heatmap - ground truth vs prediction (env {var_tag}, axes in meters)")
    #
    for var_row, var_idx in enumerate(var_pick):
        var_true_pos_m = normalized_to_meters(var_true_sets[var_idx])
        var_pred_pos_m = normalized_to_meters(var_pred_sets[var_idx])
        var_panels = ((var_true_sets[var_idx], var_true_pos_m, "ground truth"),
                      (var_pred_sets[var_idx], var_pred_pos_m, "prediction"))
        for var_col, (var_set, var_pos_m, var_title) in enumerate(var_panels):
            var_ax = var_axes[var_row][var_col]
            var_ax.imshow(splat_positions(var_set, var_grid_size, var_sigma), origin="lower",
                          cmap="viridis", vmin=0, vmax=1, extent=var_extent_m, aspect="equal")
            var_ax.set_xticks(np.arange(0, METERS_PER_UNIT + 1, 2))
            var_ax.set_yticks(np.arange(0, METERS_PER_UNIT + 1, 2))
            var_ax.set_xlabel("x (m)")
            var_ax.set_ylabel("y (m)")
            if len(var_pos_m):
                var_ax.scatter(var_pos_m[:, 0], var_pos_m[:, 1],
                               s=90, facecolors="none", edgecolors="red", linewidths=1.5)
                for var_pos in var_pos_m:
                    var_ax.annotate(f"({var_pos[0]:.2f}, {var_pos[1]:.2f}) m",
                                    (var_pos[0], var_pos[1]), textcoords="offset points",
                                    xytext=(6, 6), fontsize=7, color="red")
            var_ax.set_title(f"{var_title} - {len(var_pos_m)} "
                             + ("point" if len(var_pos_m) == 1 else "points"))
        #
        print(f"  [sample {var_idx}] true {_format_positions_meters(var_true_pos_m)} m | "
              f"pred {_format_positions_meters(var_pred_pos_m)} m")
    #
    var_fig.tight_layout(rect=[0, 0, 1, 0.96])
    var_out_path = os.path.join(save_dir, f"coords_heatmap_{var_tag}.png")
    var_fig.savefig(var_out_path, dpi=120)
    plt.close(var_fig)
    return var_out_path


def _plot_coord_curves(var_history, var_save_path, var_tag="coord"):
    """
    [description]
    : write the coordinate-head training history (loss, balanced count accuracy, count MAE, position
      error, and the raw vs EMA-smoothed count-decision thresholds) to a PNG and a CSV after every
      epoch. The threshold panel is what makes the per-epoch jitter of the calibrated thresholds, and
      the smoothing that damps it, visible.
    """
    if not var_save_path or not var_history.get("epoch"):
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(str(var_save_path), exist_ok=True)
    var_panels = [("loss", "loss", [("train_loss", "train")]),
                  ("valid balanced count accuracy", "balanced acc",
                   [("valid_bal", "raw"), ("valid_bal_smoothed", "smoothed (EMA)")]),
                  ("valid count MAE", "MAE",
                   [("valid_mae", "raw"), ("valid_mae_smoothed", "smoothed (EMA)")]),
                  ("valid position error (m)", "m", [("valid_error_m", "raw")]),
                  ("count decision thresholds", "probability",
                   [("valid_threshold", "objectness (raw)"),
                    ("valid_threshold_smoothed", "objectness (EMA)"),
                    ("valid_empty_threshold", "empty gate (raw)"),
                    ("valid_empty_threshold_smoothed", "empty gate (EMA)")])]
    var_fig, var_axes = plt.subplots(1, len(var_panels), figsize=(25, 4))
    for var_ax, (var_title, var_ylabel, var_series) in zip(var_axes, var_panels):
        for var_key, var_label in var_series:
            var_ax.plot(var_history["epoch"], var_history[var_key], label=var_label)
        var_ax.set_title(var_title)
        var_ax.set_xlabel("epoch")
        var_ax.set_ylabel(var_ylabel)
        var_ax.grid(alpha=0.3)
        var_ax.legend(fontsize=8)
    var_fig.suptitle(f"{var_tag} - coordinate head")
    var_fig.tight_layout(rect=[0, 0, 1, 0.94])
    var_png = os.path.join(str(var_save_path), f"{var_tag}_curves.png")
    var_fig.savefig(var_png, dpi=110)
    plt.close(var_fig)
    var_csv = os.path.join(str(var_save_path), f"{var_tag}_curves.csv")
    var_keys = ["epoch"] + [var_key for _, _, var_series in var_panels for var_key, _ in var_series]
    with open(var_csv, "w", newline="") as var_file:
        var_writer = csv.writer(var_file)
        var_writer.writerow(var_keys)
        for var_row in range(len(var_history["epoch"])):
            var_writer.writerow([var_history[var_key][var_row] for var_key in var_keys])
    return var_png, var_csv


def train_coord(model, optimizer, data_train_set, data_valid_set, var_batch_size, var_epochs, device,
                patience=150, var_save_dir=None, var_tag="coord"):
    """
    [description]
    : train the coordinate head with the Hungarian set loss (shared with SetLocalizer). The count uses
      the same two-stage decision as the density head: an empty gate on the maximum objectness
      probability, then the number of queries whose objectness clears the stage-2 threshold. Both
      thresholds are resolved on the validation split each epoch (or pinned by
      preset["density"]["eval_threshold"] / ["empty_threshold"]), smoothed with an EMA, and the epoch's
      metric is computed with the smoothed values so a threshold re-fit on a small validation batch
      cannot make model selection track its jitter. The checkpoint is selected on the EMA-smoothed
      balanced count accuracy, so model selection tracks the reported locate-then-count rule.
    : data_train_set / data_valid_set: TensorDataset of (CSI, positions (K,2) normalized, mask (K,))
    : return: best state dict
    """
    from src.models.set_localization import hungarian_set_loss
    ## Class balance, the same measure the density head applies (preset["density"]["balance_empty_class"]):
    ## WiMANS has only 5.3% count-0 frames per room (99 of 1881), so a plain shuffle lets the set head
    ## ignore the empty case. Draw every count class equally often instead, without discarding data.
    if preset["density"].get("balance_empty_class", False):
        var_sample_weights = set_count_sampling_weights(data_train_set)
        var_sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(var_sample_weights, dtype=torch.double),
            num_samples=len(data_train_set), replacement=True)
        var_train_loader = torch.utils.data.DataLoader(
            data_train_set, var_batch_size, sampler=var_sampler, pin_memory=True)
    else:
        var_train_loader = torch.utils.data.DataLoader(
            data_train_set, var_batch_size, shuffle=True, pin_memory=True)
    var_valid_loader = torch.utils.data.DataLoader(data_valid_set, len(data_valid_set))
    var_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=preset["nn"]["scheduler"]["num_warmup_epochs"] * len(var_train_loader),
        num_training_steps=var_epochs * len(var_train_loader),
        min_lr_ratio=preset["nn"]["scheduler"]["min_lr_ratio"])
    #
    var_best_score, var_best_mae, var_best_weight, var_counter = -np.inf, np.inf, None, 0
    var_ema_bal, var_ema_mae, var_ema_decay = None, None, 0.3
    var_ema_threshold, var_ema_empty_threshold = None, None
    var_history = {var_key: [] for var_key in ("epoch", "train_loss", "valid_bal", "valid_bal_smoothed",
                                               "valid_mae", "valid_mae_smoothed", "valid_error_m",
                                               "valid_threshold", "valid_threshold_smoothed",
                                               "valid_empty_threshold", "valid_empty_threshold_smoothed")}
    for var_epoch in range(var_epochs):
        model.train()
        var_loss_sum, var_batches = 0.0, 0
        for var_batch in var_train_loader:
            var_x, var_pos, var_mask = var_batch[0].to(device), var_batch[1].to(device), var_batch[2].to(device)
            var_xy, var_obj = model(augment_csi(var_x))
            var_loss = hungarian_set_loss(var_xy, var_obj, var_pos, var_mask)
            optimizer.zero_grad()
            var_loss.backward()
            optimizer.step()
            var_scheduler.step()
            var_loss_sum += float(var_loss.detach())
            var_batches += 1
        var_train_loss = var_loss_sum / max(1, var_batches)
        #
        model.eval()
        with torch.no_grad():
            var_batch = next(iter(var_valid_loader))
            var_xy, var_obj = model(var_batch[0].to(device))
        var_true_count = var_batch[2].sum(axis=1).round().numpy()
        var_empty_threshold, var_threshold = resolve_coord_count_decision(
            var_obj.cpu().numpy(), var_true_count)
        ## Smooth both thresholds: a threshold re-fit on a small validation batch every epoch jitters,
        ## and the metric that selects the checkpoint must not chase that jitter. The epoch's metric is
        ## computed with the smoothed values; the raw ones are still logged. _ema_update ignores the
        ## degenerate +-inf empty-gate values (no empty or no occupied frame in the split).
        var_ema_threshold = _ema_update(var_ema_threshold, var_threshold, var_ema_decay)
        var_ema_empty_threshold = _ema_update(
            var_ema_empty_threshold, _finite_or_nan(var_empty_threshold), var_ema_decay)
        var_metrics = coord_metrics(var_xy.cpu().numpy(), var_obj.cpu().numpy(),
                                    var_batch[1].numpy(), var_batch[2].numpy(),
                                    var_ema_threshold, var_ema_empty_threshold)
        var_ema_bal = _ema_update(var_ema_bal, var_metrics["balanced_accuracy"], var_ema_decay)
        var_ema_mae = _ema_update(var_ema_mae, var_metrics["mae"], var_ema_decay)
        var_empty_threshold_log = _finite_or_nan(var_empty_threshold)
        wandb.log({"epoch": var_epoch, "train_loss": var_train_loss,
                   "valid_balanced_accuracy": var_metrics["balanced_accuracy"],
                   "valid_balanced_accuracy_smoothed": var_ema_bal,
                   "valid_mae": var_metrics["mae"], "valid_mae_smoothed": var_ema_mae,
                   "valid_position_error_m": var_metrics["position_error_m"],
                   "valid_objectness_threshold": var_threshold,
                   "valid_objectness_threshold_smoothed": var_ema_threshold,
                   "valid_empty_threshold": var_empty_threshold_log,
                   "valid_empty_threshold_smoothed": var_ema_empty_threshold,
                   "learning_rate": optimizer.param_groups[0]["lr"]})
        print(f"Epoch {var_epoch}/{var_epochs} - Loss %.4f - Valid Bal %.4f - Valid MAE %.4f - "
              "PosErr %.3f m - Thr %.2f (smooth %.2f) - EmptyThr %.3f (smooth %.3f)"
              % (var_train_loss, var_metrics["balanced_accuracy"], var_metrics["mae"],
                 var_metrics["position_error_m"], var_threshold, var_ema_threshold,
                 var_empty_threshold_log, var_ema_empty_threshold))
        var_history["epoch"].append(var_epoch)
        var_history["train_loss"].append(var_train_loss)
        var_history["valid_bal"].append(var_metrics["balanced_accuracy"])
        var_history["valid_bal_smoothed"].append(var_ema_bal)
        var_history["valid_mae"].append(var_metrics["mae"])
        var_history["valid_mae_smoothed"].append(var_ema_mae)
        var_history["valid_error_m"].append(var_metrics["position_error_m"])
        var_history["valid_threshold"].append(var_threshold)
        var_history["valid_threshold_smoothed"].append(var_ema_threshold)
        var_history["valid_empty_threshold"].append(var_empty_threshold_log)
        var_history["valid_empty_threshold_smoothed"].append(var_ema_empty_threshold)
        _plot_coord_curves(var_history, var_save_dir, var_tag)
        #
        if (var_ema_bal > var_best_score
                or (var_ema_bal == var_best_score and var_ema_mae < var_best_mae)):
            var_best_score, var_best_mae = var_ema_bal, var_ema_mae
            var_best_weight = copy.deepcopy(model.state_dict())
            var_counter = 0
        else:
            var_counter += 1
        if var_counter >= patience:
            print(f"Early stopping triggered at epoch {var_epoch}")
            break
    if var_best_weight is None:
        var_best_weight = copy.deepcopy(model.state_dict())
    print(f"Best smoothed balanced count accuracy: {var_best_score:.6f}, smoothed count MAE: {var_best_mae:.6f}")
    return var_best_weight


def run_density_map_coords(train_sets_by_env, test_sets_by_env, var_repeat=3,
                           save_path="./visualizations/coords", var_env=None):
    """
    [description]
    : run the coordinate-regression head (a single head that directly regresses user locations). It
      trains on every room in train_sets_by_env and evaluates on every room in test_sets_by_env; pass
      one training room for the single-room protocol or two for leave-one-room-out. The count is the
      number of queries whose objectness clears the validation-calibrated threshold, and the predicted
      locations are reported in meters, so count and location come from the same head.
    : train_sets_by_env: dict, env name -> (CSI amplitude, occupancy targets)
    : test_sets_by_env: dict, env name -> (CSI amplitude, occupancy targets)
    : var_repeat: int, number of repeated experiments
    : var_env: optional env name used only for the run name (single-room dispatch)
    : return: dict, env name -> averaged count/location metrics with SE
    """
    from src.models.set_localization import SetLocalizer, select_sets
    device = select_device()
    print(f"Using device: {device}")
    var_num_queries = preset["nn"]["num_obj_queries"]
    var_train_envs = list(train_sets_by_env)
    #
    var_train_sets, var_valid_sets = [], []
    for var_name in var_train_envs:
        var_x, var_y = train_sets_by_env[var_name]
        var_x = np.asarray(var_x, dtype=np.float32).reshape(len(var_x), var_x.shape[1], -1)
        var_pos, var_mask = occupancy_to_position_batch(var_y, preset["layouts"][var_name])
        var_ds = TensorDataset(torch.from_numpy(var_x), torch.from_numpy(var_pos), torch.from_numpy(var_mask))
        var_perm = np.random.RandomState(39).permutation(len(var_ds))
        var_n_valid = max(1, int(round(0.1 * len(var_ds))))
        var_valid_sets.append(torch.utils.data.Subset(var_ds, var_perm[:var_n_valid].tolist()))
        var_train_sets.append(torch.utils.data.Subset(var_ds, var_perm[var_n_valid:].tolist()))
    var_train_set = torch.utils.data.ConcatDataset(var_train_sets)
    var_valid_set = torch.utils.data.ConcatDataset(var_valid_sets)
    var_x_shape = tuple(var_train_sets[0].dataset.tensors[0].shape[1:])
    print(f"Training rooms: {var_train_envs} ({len(var_train_set)} train / {len(var_valid_set)} valid) | "
          f"Test rooms: {list(test_sets_by_env)}")
    #
    var_test = {}
    for var_name, (var_x, var_y) in test_sets_by_env.items():
        var_x = np.asarray(var_x, dtype=np.float32).reshape(len(var_x), var_x.shape[1], -1)
        var_pos, var_mask = occupancy_to_position_batch(var_y, preset["layouts"][var_name])
        var_test[var_name] = (torch.from_numpy(var_x), torch.from_numpy(var_pos), torch.from_numpy(var_mask))
    #
    var_env_metrics = {}
    for var_r in range(var_repeat):
        print("Repeat", var_r)
        wandb.init(project="density_map_coords",
                   name=f"Coord{var_r}_" + "_".join(var_train_envs), config=preset, reinit=True)
        torch.random.manual_seed(var_r + 39)
        #
        model = SetLocalizer(var_x_shape, embedding_dim=100, num_queries=var_num_queries).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=preset["nn"]["lr"],
                                     weight_decay=preset["nn"]["weight_decay"])
        var_best_weight = train_coord(model, optimizer, var_train_set, var_valid_set,
                                      preset["nn"]["batch_size"], preset["nn"]["epoch"], device,
                                      patience=preset["density"].get("patience", 150),
                                      var_save_dir=save_path, var_tag=f"coord_r{var_r}")
        model.load_state_dict(var_best_weight)
        model.eval()
        #
        ## two-stage count decision from the training rooms' validation split: the empty gate and the
        ## objectness threshold (or the pinned preset values). Both are reported so a run is
        ## self-describing.
        with torch.no_grad():
            var_vx, var_vpos, var_vmask = next(iter(
                torch.utils.data.DataLoader(var_valid_set, len(var_valid_set))))
            var_vxy, var_vobj = model(var_vx.to(device))
        var_empty_threshold, var_threshold = resolve_coord_count_decision(
            var_vobj.cpu().numpy(), var_vmask.sum(axis=1).round().numpy())
        #
        for var_name, (var_x, var_pos, var_mask) in var_test.items():
            with torch.no_grad():
                var_xy, var_obj = model(var_x.to(device))
            var_metrics = coord_metrics(var_xy.cpu().numpy(), var_obj.cpu().numpy(),
                                        var_pos.numpy(), var_mask.numpy(), var_threshold,
                                        var_empty_threshold)
            var_metrics["threshold"] = var_threshold
            var_metrics["empty_threshold"] = _finite_or_nan(var_empty_threshold)
            var_env_metrics.setdefault(var_name, []).append(var_metrics)
            wandb.log({f"test_results_per_env/{var_name}/accuracy": var_metrics["accuracy"],
                       f"test_results_per_env/{var_name}/balanced_accuracy": var_metrics["balanced_accuracy"],
                       f"test_results_per_env/{var_name}/mae": var_metrics["mae"],
                       f"test_results_per_env/{var_name}/position_error_m": var_metrics["position_error_m"],
                       f"test_results_per_env/{var_name}/threshold": var_threshold,
                       f"test_results_per_env/{var_name}/empty_threshold":
                           _finite_or_nan(var_empty_threshold)}, step=var_r + 100000)
            print(f"  [{var_name}] COUNT Acc {var_metrics['accuracy']:.4f} - "
                  f"Bal {var_metrics['balanced_accuracy']:.4f} - MAE {var_metrics['mae']:.4f} - "
                  f"PosErr {var_metrics['position_error_m']:.3f} m "
                  f"(miss {var_metrics['position_miss_rate']:.3f}, "
                  f"fp {var_metrics['position_false_positive_rate']:.3f}) - "
                  f"Thr {var_threshold:.2f} - EmptyThr {_finite_or_nan(var_empty_threshold):.3f}")
            #
            if var_r == var_repeat - 1:
                ## heatmap analogue of the density figure: rasterize the selected query set and the
                ## ground-truth positions onto the same grid and save it next to the curves. The empty
                ## gate is applied to the rendered sets too, so the figure shows the same decision the
                ## metrics score.
                var_pred_sets = select_sets(var_xy, var_obj, var_threshold)
                if var_empty_threshold is not None:
                    var_obj_max = coord_objectness_maxima(var_obj.cpu())
                    var_pred_sets = [var_set if var_obj_max[var_i] >= var_empty_threshold
                                     else np.zeros((0, 2), dtype=var_set.dtype)
                                     for var_i, var_set in enumerate(var_pred_sets)]
                var_pos_np, var_mask_np = var_pos.numpy(), var_mask.numpy()
                var_true_sets = [var_pos_np[var_i][var_mask_np[var_i] > 0.5]
                                 for var_i in range(len(var_pos_np))]
                var_fig_path = visualize_coord_heatmap(var_true_sets, var_pred_sets, save_path,
                                                       var_tag=var_name)
                print(f"  Figure saved to {var_fig_path}")
        #
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    #
    var_num_classes = preset["nn"]["num_count_classes"]
    var_metric_names = ("accuracy", "balanced_accuracy", "mae", "occupancy_accuracy", "occupancy_f1",
                        "position_error_m", "position_miss_rate", "position_false_positive_rate")
    results = {}
    for var_name, var_rep_list in var_env_metrics.items():
        var_env_result = {}
        for var_metric in var_metric_names:
            var_arr = np.array([var_rep[var_metric] for var_rep in var_rep_list], dtype=float)
            if np.isfinite(var_arr).any():
                var_env_result[f"avg_{var_metric}"] = float(np.nanmean(var_arr))
                var_std = float(np.nanstd(var_arr, ddof=1)) if len(var_arr) > 1 else 0.0
            else:
                ## e.g. the occupancy metrics: the coordinate head has no per-location readout
                var_env_result[f"avg_{var_metric}"] = float("nan")
                var_std = 0.0
            var_env_result[f"std_{var_metric}"] = var_std
            var_env_result[f"se_{var_metric}"] = var_std / np.sqrt(len(var_arr)) if len(var_arr) > 1 else 0.0
        var_env_result["avg_threshold"] = float(np.mean([var_rep["threshold"] for var_rep in var_rep_list]))
        var_empty_values = np.array([var_rep["empty_threshold"] for var_rep in var_rep_list])
        var_env_result["avg_empty_threshold"] = (float(np.nanmean(var_empty_values))
                                                 if np.isfinite(var_empty_values).any() else float("nan"))
        ## the layout-free peak rule is derived too: record it next to the metrics it produced
        for var_key in ("peak_threshold", "per_person_mass"):
            var_values = np.array([var_rep[var_key] for var_rep in var_rep_list], dtype=float)
            var_env_result[f"avg_{var_key}"] = (float(np.nanmean(var_values))
                                                if np.isfinite(var_values).any() else float("nan"))
        var_env_result["per_class_accuracy"] = {
            var_class: float(np.mean([var_rep["per_class_accuracy"][var_class] for var_rep in var_rep_list]))
            for var_class in range(var_num_classes)}
        results[var_name] = var_env_result
        print(f"\n[{var_name}] avg over {var_repeat} repeats: "
              f"count Acc {var_env_result['avg_accuracy']:.4f} ± {var_env_result['se_accuracy']:.4f} | "
              f"MAE {var_env_result['avg_mae']:.4f} ± {var_env_result['se_mae']:.4f} | "
              f"PosErr {var_env_result['avg_position_error_m']:.4f} ± "
              f"{var_env_result['se_position_error_m']:.4f} m")
    wandb.finish()
    return results
