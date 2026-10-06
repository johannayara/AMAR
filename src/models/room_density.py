"""
[file]          room_density.py
[description]   Multi-user room-agnostic localization via the continuous density head.

                Each sample's label is a SET of positions (WiMANS: one per occupied user, H-WILD:
                the single person's UWB position). The density head predicts an occupancy map over
                the room's shared [0, 1]^2 frame; the training target is the sum of peak-normalized
                Gaussians at the label positions, clamped to [0, 1], so several users render as
                several peaks. At evaluation the predicted map's peaks are matched to the ground-truth
                set with a Hungarian assignment, which keeps the multi-user structure intact: the
                reported error is the mean matched distance plus a penalty for missed and spurious
                positions, alongside count MAE and detection precision/recall.

                This is the density-map formulation of the WiMANS cross-domain LOR
                (src/models/density_map.py) generalised from fixed per-room location kernels to
                arbitrary position sets, which is what lets H-WILD's continuous positions and WiMANS'
                discrete multi-user locations share one model and one frame.
"""
#
##

import copy
import gc
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import center_of_mass, label as nd_label, maximum as nd_maximum, maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, TensorDataset

from configs.preset import preset
from src.models.density_map import DensityMapNet, get_cosine_schedule_with_warmup, augment_csi
from src.utils import select_device, save_run_outputs, build_run_stem, join_envs, run_timestamp
import wandb

#
##


def render_multi_targets(var_pos, var_mask, var_grid, var_sigma):
    """
    [description]
    : render a batch of multi-user density targets: the sum of one peak-normalized Gaussian per valid
      label position, clamped to [0, 1] so it is a valid Bernoulli target.
    : var_pos: tensor (batch, max_users, 2) normalized positions
    : var_mask: tensor (batch, max_users) with 1.0 on valid positions
    : return: tensor (batch, grid, grid)
    """
    #
    var_axis = (torch.arange(var_grid, dtype=torch.float32, device=var_pos.device) + 0.5) / var_grid
    var_yy, var_xx = torch.meshgrid(var_axis, var_axis, indexing="ij")
    var_cx = var_pos[..., 0].unsqueeze(-1).unsqueeze(-1)
    var_cy = var_pos[..., 1].unsqueeze(-1).unsqueeze(-1)
    var_blob = torch.exp(-((var_xx - var_cx) ** 2 + (var_yy - var_cy) ** 2) / (2 * var_sigma ** 2))
    var_peak = var_blob.flatten(2).max(2).values.clamp_min(1e-12)
    var_blob = var_blob / var_peak.unsqueeze(-1).unsqueeze(-1)
    return (var_blob * var_mask.unsqueeze(-1).unsqueeze(-1)).sum(dim=1).clamp(0.0, 1.0)


def extract_peak_sets(var_density, var_threshold_frac, var_abs_floor=0.0, var_max_peaks=None):
    """
    [description]
    : per-sample sets of predicted positions, one set per density map. A cell is a peak when it is
      the maximum of its 3x3 neighbourhood and clears max(threshold_frac * map_max, abs_floor); the
      peaks are then ranked by their map value and the top var_max_peaks are kept. The cap matters:
      the task has at most var_max_peaks users, so an un-capped extraction on a not-yet-concentrated
      map returns dozens of spurious maxima and makes the count error meaningless.
    : var_max_peaks: int or None, default preset["nn"]["num_obj_queries"]
    : return: list of (num_peaks, 2) numpy arrays, at most var_max_peaks entries each
    """
    #
    if var_max_peaks is None:
        var_max_peaks = preset["nn"]["num_obj_queries"]
    var_arr = var_density.detach().cpu().numpy() if torch.is_tensor(var_density) else np.asarray(var_density)
    var_sets = []
    for var_idx in range(len(var_arr)):
        var_map = var_arr[var_idx]
        var_max = float(var_map.max())
        if var_max <= 0 or var_max < var_abs_floor:
            var_sets.append(np.zeros((0, 2), dtype=np.float32))
            continue
        var_cut = max(var_threshold_frac * var_max, var_abs_floor)
        var_mask = (var_map == maximum_filter(var_map, size=3)) & (var_map > var_cut)
        var_labels, var_num = nd_label(var_mask)
        if var_num == 0:
            var_sets.append(np.zeros((0, 2), dtype=np.float32))
            continue
        var_coms = center_of_mass(var_map, var_labels, range(1, var_num + 1))
        var_values = nd_maximum(var_map, var_labels, range(1, var_num + 1))
        var_order = np.argsort(var_values)[::-1][:var_max_peaks]
        var_h, var_w = var_map.shape
        var_sets.append(np.array([[var_coms[j][1] / var_w, var_coms[j][0] / var_h] for j in var_order],
                                 dtype=np.float32))
    return var_sets


def set_localization_metrics(var_pred_sets, var_true_sets, var_span, var_match_threshold_m=1.5):
    """
    [description]
    : multi-user localization error between predicted and true position sets. Each sample's predicted
      set is matched to its true set with a Hungarian assignment; the sample error is the mean matched
      distance plus a per-miss penalty (half the room diagonal) for unmatched true and spurious
      predicted positions, divided by the larger set size. Also reports the count and detection
      statistics.
    : var_pred_sets / var_true_sets: lists of (k, 2) normalized position arrays
    : var_span: (x_span, y_span) in meters, or (N, 2) when the set mixes rooms
    : var_match_threshold_m: distance below which a matched pair counts as a true positive
    : return: dict of mean error (m), matched MDE (m), count MAE, exact-count accuracy, precision,
      recall, F1
    """
    #
    var_num = len(var_pred_sets)
    var_span = np.asarray(var_span, dtype=np.float64)
    if var_span.ndim == 1:
        var_span = np.tile(var_span, (var_num, 1))
    #
    var_errs, var_matched, var_true_counts, var_pred_counts = [], [], [], []
    var_tp = var_fp = var_fn = 0
    for var_idx in range(var_num):
        var_pred = np.asarray(var_pred_sets[var_idx], dtype=np.float64)
        var_true = np.asarray(var_true_sets[var_idx], dtype=np.float64)
        var_pred_m = var_pred * var_span[var_idx]
        var_true_m = var_true * var_span[var_idx]
        var_num_true, var_num_pred = len(var_true_m), len(var_pred_m)
        var_true_counts.append(var_num_true)
        var_pred_counts.append(var_num_pred)
        var_penalty = 0.5 * float(np.hypot(var_span[var_idx, 0], var_span[var_idx, 1]))
        #
        if var_num_pred and var_num_true:
            var_cost = cdist(var_pred_m, var_true_m)
            var_rows, var_cols = linear_sum_assignment(var_cost)
            var_dist = var_cost[var_rows, var_cols]
            var_matched.extend(var_dist.tolist())
            var_near = var_dist <= var_match_threshold_m
            var_tp += int(var_near.sum())
            var_fp += int((~var_near).sum()) + (var_num_pred - len(var_dist))
            var_fn += int((~var_near).sum()) + (var_num_true - len(var_dist))
            var_errs.append((var_dist.sum()
                             + var_penalty * ((var_num_true - len(var_dist)) + (var_num_pred - len(var_dist))))
                            / max(var_num_true, var_num_pred))
        elif var_num_pred == 0 and var_num_true == 0:
            var_errs.append(0.0)
        else:
            var_fp += var_num_pred
            var_fn += var_num_true
            var_errs.append(var_penalty)
    #
    var_precision = var_tp / (var_tp + var_fp) if (var_tp + var_fp) > 0 else 0.0
    var_recall = var_tp / (var_tp + var_fn) if (var_tp + var_fn) > 0 else 0.0
    var_f1 = 2 * var_precision * var_recall / (var_precision + var_recall) if (var_precision + var_recall) > 0 else 0.0
    var_true_counts = np.asarray(var_true_counts)
    var_pred_counts = np.asarray(var_pred_counts)
    #
    return {
        "mean_error_m": float(np.mean(var_errs)),
        "mde_matched_m": float(np.mean(var_matched)) if var_matched else float("nan"),
        "count_mae": float(np.abs(var_true_counts - var_pred_counts).mean()),
        "exact_count_acc": float((var_true_counts == var_pred_counts).mean()),
        "detection_precision": float(var_precision),
        "detection_recall": float(var_recall),
        "detection_f1": float(var_f1),
    }


def _resolve_spans(var_rooms, var_spans):
    """
    [description]
    : per-room (x_span, y_span) in meters. Defaults to the H-WILD geometry when not given.
    """
    if var_spans is not None:
        return var_spans
    from src.data.hwild import room_span
    return {var_room: room_span(var_room) for var_room in var_rooms}


def _resolve_bounds(var_rooms, var_bounds):
    """
    [description]
    : per-room (x_min, x_max, y_min, y_max) in meters. Defaults to the H-WILD geometry when not given.
    """
    if var_bounds is not None:
        return var_bounds
    from src.data.hwild import HWILD_ROOMS
    return {var_room: (HWILD_ROOMS[var_room]["x_range"][0], HWILD_ROOMS[var_room]["x_range"][1],
                       HWILD_ROOMS[var_room]["y_range"][0], HWILD_ROOMS[var_room]["y_range"][1])
            for var_room in var_rooms}


def _true_sets(var_pos, var_mask):
    """
    [description]
    : list of per-sample true position sets from a padded (pos, mask) batch.
    """
    var_pos = var_pos.detach().cpu().numpy()
    var_mask = var_mask.detach().cpu().numpy()
    return [var_pos[var_idx][var_mask[var_idx] > 0.5] for var_idx in range(len(var_pos))]


def _plot_curves(var_history, var_save_dir, var_tag="training"):
    """
    [description]
    : rewrite the training/validation loss and validation localization-error curves after an epoch.
    """
    #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    #
    os.makedirs(str(var_save_dir), exist_ok=True)
    var_fig, var_axes = plt.subplots(1, 2, figsize=(11, 4))
    var_axes[0].plot(var_history["epoch"], var_history["train_loss"], label="train")
    var_axes[0].plot(var_history["epoch"], var_history["valid_loss"], label="valid")
    var_axes[0].set_xlabel("epoch"); var_axes[0].set_ylabel("BCE loss"); var_axes[0].legend()
    var_axes[1].plot(var_history["epoch"], var_history["valid_error_m"], label="valid set error (m)")
    var_axes[1].plot(var_history["epoch"], var_history["valid_error_m_smoothed"], label="smoothed")
    var_axes[1].set_xlabel("epoch"); var_axes[1].set_ylabel("error (m)"); var_axes[1].legend()
    var_fig.suptitle(f"{var_tag} - multi-user density localization")
    var_fig.tight_layout(rect=[0, 0, 1, 0.95])
    var_path = os.path.join(str(var_save_dir), f"{var_tag}_curves.png")
    var_fig.savefig(var_path, dpi=110)
    plt.close(var_fig)
    return var_path


def train_room_density(model, optimizer, data_train_set, data_valid_set, var_batch_size, var_epochs,
                       device, var_span_bank, var_grid, var_sigma, patience=60,
                       var_save_dir=None, var_tag="training"):
    """
    [description]
    : train the density head as a multi-user localizer. The objective is the BCE between the predicted
      map and the rendered multi-user target; model selection tracks the EMA-smoothed validation set
      error in meters, so the checkpoint is the best localizer.
    : data_train_set / data_valid_set: TensorDataset of (CSI window, positions, mask, room index)
    : var_span_bank: tensor (num_rooms, 2) of (x_span, y_span), one row per training room
    : return: (best state dict, history dict)
    """
    #
    var_train_loader = DataLoader(data_train_set, var_batch_size, shuffle=True, pin_memory=True)
    var_valid_loader = DataLoader(data_valid_set, len(data_valid_set))
    var_peak_threshold = preset["density"]["peak_threshold"]
    #
    var_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=preset["nn"]["scheduler"]["num_warmup_epochs"] * len(var_train_loader),
        num_training_steps=var_epochs * len(var_train_loader),
        min_lr_ratio=preset["nn"]["scheduler"]["min_lr_ratio"])
    #
    var_history = {var_key: [] for var_key in (
        "epoch", "train_loss", "valid_loss", "valid_error_m", "valid_error_m_smoothed", "learning_rate")}
    var_best_score, var_best_weight, var_epoch_saved, var_counter = np.inf, None, 0, 0
    var_ema_error, var_ema_decay = None, 0.3
    #
    for var_epoch in range(var_epochs):
        var_time = time.time()
        model.train()
        var_loss_sum, var_batches = 0.0, 0
        for var_batch in var_train_loader:
            var_x = var_batch[0].to(device)
            var_pos, var_mask = var_batch[1].to(device), var_batch[2].to(device)
            var_x = augment_csi(var_x)
            _, _, var_logits = model(var_x)
            var_target = render_multi_targets(var_pos, var_mask, var_grid, var_sigma)
            var_loss = F.binary_cross_entropy_with_logits(var_logits, var_target)
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
            var_valid_batch = next(iter(var_valid_loader))
            var_valid_x = var_valid_batch[0].to(device)
            var_valid_pos, var_valid_mask = var_valid_batch[1], var_valid_batch[2]
            var_valid_density, _, var_valid_logits = model(var_valid_x)
            var_valid_target = render_multi_targets(var_valid_pos.to(device), var_valid_mask.to(device),
                                                    var_grid, var_sigma)
            var_valid_loss = float(F.binary_cross_entropy_with_logits(
                var_valid_logits, var_valid_target).detach())
        var_span = var_span_bank[var_valid_batch[3]].numpy()
        var_metrics = set_localization_metrics(
            extract_peak_sets(var_valid_density, var_peak_threshold),
            _true_sets(var_valid_pos, var_valid_mask), var_span)
        var_ema_error = (var_metrics["mean_error_m"] if var_ema_error is None
                         else var_ema_decay * var_metrics["mean_error_m"] + (1 - var_ema_decay) * var_ema_error)
        #
        wandb.log({"epoch": var_epoch, "train_loss": var_train_loss, "valid_loss": var_valid_loss,
                   "valid_error_m": var_metrics["mean_error_m"],
                   "valid_error_m_smoothed": var_ema_error,
                   "valid_count_mae": var_metrics["count_mae"],
                   "valid_detection_f1": var_metrics["detection_f1"],
                   "learning_rate": optimizer.param_groups[0]["lr"]})
        print(f"Epoch {var_epoch}/{var_epochs} - %.2fs" % (time.time() - var_time),
              "- Loss %.6f" % var_train_loss, "- ValidLoss %.6f" % var_valid_loss,
              "- ValidErr %.4f m" % var_metrics["mean_error_m"],
              "- CntMAE %.3f" % var_metrics["count_mae"],
              "- DetF1 %.3f" % var_metrics["detection_f1"], "- Smooth %.4f m" % var_ema_error)
        #
        for var_key, var_value in (("epoch", var_epoch), ("train_loss", var_train_loss),
                                   ("valid_loss", var_valid_loss),
                                   ("valid_error_m", var_metrics["mean_error_m"]),
                                   ("valid_error_m_smoothed", var_ema_error),
                                   ("learning_rate", optimizer.param_groups[0]["lr"])):
            var_history[var_key].append(var_value)
        if var_save_dir is not None:
            _plot_curves(var_history, var_save_dir, var_tag)
        #
        if var_ema_error < var_best_score:
            var_best_score, var_best_weight, var_epoch_saved, var_counter = \
                var_ema_error, copy.deepcopy(model.state_dict()), var_epoch, 0
        else:
            var_counter += 1
        if var_counter >= patience:
            print(f"Early stopping triggered at epoch {var_epoch}")
            break
    #
    if var_best_weight is None:
        var_best_weight = copy.deepcopy(model.state_dict())
    print(f"Epoch that the model was saved {var_epoch_saved} - best smoothed error {var_best_score:.4f} m")
    return var_best_weight, var_history


def visualize_room_density(var_true_sets, var_pred_sets, var_room, save_dir, var_bounds):
    """
    [description]
    : scatter every true (green) and predicted (red) position of a test room in meters.
    """
    #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    #
    os.makedirs(save_dir, exist_ok=True)
    var_x0, var_x1, var_y0, var_y1 = var_bounds
    var_true = np.concatenate([np.asarray(s) for s in var_true_sets if len(s)]) if any(len(s) for s in var_true_sets) else np.zeros((0, 2))
    var_pred = np.concatenate([np.asarray(s) for s in var_pred_sets if len(s)]) if any(len(s) for s in var_pred_sets) else np.zeros((0, 2))
    var_true_m = np.stack([var_x0 + var_true[:, 0] * (var_x1 - var_x0),
                           var_y0 + var_true[:, 1] * (var_y1 - var_y0)], axis=1) if len(var_true) else var_true
    var_pred_m = np.stack([var_x0 + var_pred[:, 0] * (var_x1 - var_x0),
                           var_y0 + var_pred[:, 1] * (var_y1 - var_y0)], axis=1) if len(var_pred) else var_pred
    #
    var_fig, var_ax = plt.subplots(figsize=(6, 6))
    if len(var_true_m):
        var_ax.scatter(var_true_m[:, 0], var_true_m[:, 1], s=10, c="tab:green", alpha=0.5, label="true")
    if len(var_pred_m):
        var_ax.scatter(var_pred_m[:, 0], var_pred_m[:, 1], s=10, c="tab:red", alpha=0.5, label="pred")
    var_ax.set_xlim(var_x0, var_x1); var_ax.set_ylim(var_y0, var_y1)
    var_ax.set_aspect("equal"); var_ax.set_xlabel("x (m)"); var_ax.set_ylabel("y (m)")
    var_ax.set_title(f"{var_room} - true vs predicted positions")
    var_ax.legend(loc="upper right")
    var_path = os.path.join(save_dir, f"predictions_{var_room}.png")
    var_fig.tight_layout(); var_fig.savefig(var_path, dpi=120); plt.close(var_fig)
    return var_path


def run_room_density_cross_domain(train_sets_by_room, test_sets_by_room, var_repeat=3,
                                  save_path="./visualizations/room_density",
                                  var_spans=None, var_bounds=None):
    """
    [description]
    : multi-user leave-one-room-out localization over an arbitrary room set (possibly mixing datasets).
      Trains on every room in train_sets_by_room and evaluates on every room in test_sets_by_room.
      Each training room keeps a 90/10 split for model selection; the test rooms contribute no labels.
    : train_sets_by_room / test_sets_by_room: dict room -> (X (N, window, 90), POS (N, K, 2) normalized,
      MASK (N, K))
    : return: dict, room name -> averaged multi-user localization metrics with SE
    """
    #
    device = select_device()
    print(f"Using device: {device}")
    var_grid = preset["density"]["grid_size"]
    var_sigma = preset["density"]["sigma"]
    var_peak_threshold = preset["density"]["peak_threshold"]
    var_peak_floor = preset["density"]["peak_abs_floor"]
    #
    var_all_rooms = list(train_sets_by_room) + list(test_sets_by_room)
    var_spans = _resolve_spans(var_all_rooms, var_spans)
    var_bounds = _resolve_bounds(var_all_rooms, var_bounds)
    var_train_rooms = list(train_sets_by_room)
    var_room_index = {var_room: var_idx for var_idx, var_room in enumerate(var_train_rooms)}
    var_span_bank = torch.tensor([var_spans[var_room] for var_room in var_train_rooms], dtype=torch.float32)
    #
    ## ------------------------------------------ data prep -------------------------------------------
    var_train_datasets, var_valid_datasets, var_train_pos = [], [], []
    for var_room in var_train_rooms:
        var_x_room, var_pos_room, var_mask_room = train_sets_by_room[var_room]
        var_x_room = np.asarray(var_x_room, dtype=np.float32).reshape(len(var_x_room), var_x_room.shape[1], -1)
        var_pos_room = np.asarray(var_pos_room, dtype=np.float32)
        var_mask_room = np.asarray(var_mask_room, dtype=np.float32)
        var_idx_room = np.full(len(var_x_room), var_room_index[var_room], dtype=np.int64)
        var_dataset = TensorDataset(torch.from_numpy(var_x_room), torch.from_numpy(var_pos_room),
                                    torch.from_numpy(var_mask_room), torch.from_numpy(var_idx_room))
        var_perm = np.random.RandomState(39).permutation(len(var_dataset))
        var_num_valid = max(1, int(round(0.1 * len(var_dataset))))
        var_valid_idx = np.sort(var_perm[:var_num_valid])
        var_train_idx = np.sort(var_perm[var_num_valid:])
        var_train_datasets.append(torch.utils.data.Subset(var_dataset, var_train_idx.tolist()))
        var_valid_datasets.append(torch.utils.data.Subset(var_dataset, var_valid_idx.tolist()))
        var_train_pos.append(var_pos_room[var_mask_room > 0.5])
    #
    data_train_set = torch.utils.data.ConcatDataset(var_train_datasets)
    data_valid_set = torch.utils.data.ConcatDataset(var_valid_datasets)
    var_x_shape = tuple(var_train_datasets[0].dataset.tensors[0].shape[1:])
    ## The mean training position, emitted once per sample, is the "no-information" baseline: a model
    ## that does not beat it has not learned a transferable CSI-to-position mapping.
    var_train_mean_xy = np.concatenate(var_train_pos).mean(axis=0)
    print(f"Training rooms: {var_train_rooms} ({len(data_train_set)} train / "
          f"{len(data_valid_set)} valid) | Test rooms: {list(test_sets_by_room)}")
    print(f"Constant-position baseline (mean training position): {var_train_mean_xy.round(3)}")
    #
    ## --------------------------------------- train & evaluate ---------------------------------------
    var_env_rep_metrics, var_env_last_sets = {}, {}
    for var_r in range(var_repeat):
        print("Repeat", var_r)
        wandb.init(project="room_density_localization",
                   name=f"RoomDensity{var_r}_" + "_".join(var_train_rooms), config=preset, reinit=True)
        torch.random.manual_seed(var_r + 39)
        #
        model = DensityMapNet(var_x_shape, var_layout=None, embedding_dim=100, grid_size=var_grid,
                              sigma=var_sigma).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=preset["nn"]["lr"],
                                     weight_decay=preset["nn"]["weight_decay"])
        var_best_weight, _ = train_room_density(
            model=model, optimizer=optimizer, data_train_set=data_train_set, data_valid_set=data_valid_set,
            var_batch_size=preset["nn"]["batch_size"], var_epochs=preset["nn"]["epoch"], device=device,
            var_span_bank=var_span_bank, var_grid=var_grid, var_sigma=var_sigma,
            patience=preset["density"].get("patience", 60), var_save_dir=save_path, var_tag=f"train_r{var_r}")
        model.load_state_dict(var_best_weight)
        model.eval()
        #
        for var_room, (var_x_room, var_pos_room, var_mask_room) in test_sets_by_room.items():
            var_x_room = np.asarray(var_x_room, dtype=np.float32).reshape(
                len(var_x_room), var_x_room.shape[1], -1)
            var_loader = DataLoader(TensorDataset(torch.from_numpy(var_x_room)),
                                    batch_size=preset["nn"]["batch_size"], shuffle=False)
            var_pred_sets = []
            with torch.no_grad():
                for (var_x,) in var_loader:
                    var_density, _, _ = model(var_x.to(device))
                    var_pred_sets.extend(extract_peak_sets(var_density, var_peak_threshold, var_peak_floor))
            var_true_sets = [np.asarray(var_pos_room[i])[np.asarray(var_mask_room[i]) > 0.5]
                             for i in range(len(var_pos_room))]
            var_metrics = set_localization_metrics(var_pred_sets, var_true_sets, var_spans[var_room])
            var_metrics["const_mean_error_m"] = set_localization_metrics(
                [np.tile(var_train_mean_xy, (1, 1)) for _ in range(len(var_true_sets))],
                var_true_sets, var_spans[var_room])["mean_error_m"]
            var_env_rep_metrics.setdefault(var_room, []).append(var_metrics)
            var_env_last_sets[var_room] = (var_true_sets, var_pred_sets)
            #
            wandb.log({f"test_results_per_env/{var_room}/mean_error_m": var_metrics["mean_error_m"],
                       f"test_results_per_env/{var_room}/count_mae": var_metrics["count_mae"],
                       f"test_results_per_env/{var_room}/detection_f1": var_metrics["detection_f1"]},
                      step=var_r + 100000)
            print(f"  [{var_room}] setErr {var_metrics['mean_error_m']:.4f} m - "
                  f"matchedMDE {var_metrics['mde_matched_m']:.4f} m - "
                  f"cntMAE {var_metrics['count_mae']:.3f} - exactCnt {var_metrics['exact_count_acc']:.3f} - "
                  f"detF1 {var_metrics['detection_f1']:.3f} - const {var_metrics['const_mean_error_m']:.4f} m")
        #
        if var_r != var_repeat - 1:
            del model, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        else:
            torch.save({"state_dict": model.state_dict(), "x_shape": var_x_shape,
                        "grid_size": var_grid, "sigma": var_sigma, "train_rooms": var_train_rooms},
                       os.path.join(save_path, "model.pth"))
    wandb.finish()
    #
    ## ---------------------------------------- aggregate ---------------------------------------------
    var_metric_names = ("mean_error_m", "mde_matched_m", "count_mae", "exact_count_acc",
                        "detection_precision", "detection_recall", "detection_f1")
    results = {}
    for var_room, var_rep_list in var_env_rep_metrics.items():
        var_env_result = {}
        for var_name in var_metric_names:
            var_arr = np.array([var_rep[var_name] for var_rep in var_rep_list], dtype=np.float64)
            var_std = float(np.nanstd(var_arr, ddof=1)) if len(var_arr) > 1 else 0.0
            var_env_result[f"avg_{var_name}"] = float(np.nanmean(var_arr))
            var_env_result[f"std_{var_name}"] = var_std
            var_env_result[f"se_{var_name}"] = var_std / np.sqrt(len(var_arr)) if len(var_arr) > 1 else 0.0
        var_env_result["avg_const_mean_error_m"] = float(
            np.mean([var_rep["const_mean_error_m"] for var_rep in var_rep_list]))
        results[var_room] = var_env_result
        var_true_sets, var_pred_sets = var_env_last_sets[var_room]
        visualize_room_density(var_true_sets, var_pred_sets, var_room, os.path.join(save_path, var_room),
                               var_bounds[var_room])
        print(f"\n[{var_room}] avg over {var_repeat} repeats: "
              f"setErr {var_env_result['avg_mean_error_m']:.4f} ± {var_env_result['se_mean_error_m']:.4f} m | "
              f"matchedMDE {var_env_result['avg_mde_matched_m']:.4f} m | "
              f"cntMAE {var_env_result['avg_count_mae']:.3f} | "
              f"detF1 {var_env_result['avg_detection_f1']:.3f} | "
              f"constant-baseline {var_env_result['avg_const_mean_error_m']:.4f} m")
    #
    return results


def format_room_density_result(var_train_rooms, var_test_rooms, results):
    """
    [description]
    : human-readable report of a multi-user leave-one-room-out density run.
    """
    #
    var_lines = ["=" * 88,
                 f"MULTI-USER ROOM-AGNOSTIC LOCALIZATION (density) - train {join_envs(var_train_rooms)}, "
                 f"test {join_envs(var_test_rooms)}",
                 "=" * 88]
    for var_room, var_res in results.items():
        var_lines.append(f"\n{var_room}:")
        var_lines.append(f"  Set error (m):        {var_res['avg_mean_error_m']:.4f} "
                         f"± {var_res['se_mean_error_m']:.4f} m (SE)")
        var_lines.append(f"  Matched MDE (m):      {var_res['avg_mde_matched_m']:.4f} "
                         f"± {var_res['se_mde_matched_m']:.4f} m (SE)")
        var_lines.append(f"  Count MAE:            {var_res['avg_count_mae']:.4f} "
                         f"± {var_res['se_count_mae']:.4f} (SE)")
        var_lines.append(f"  Exact-count accuracy: {var_res['avg_exact_count_acc']:.4f}")
        var_lines.append(f"  Detection P/R/F1:      {var_res['avg_detection_precision']:.3f} / "
                         f"{var_res['avg_detection_recall']:.3f} / {var_res['avg_detection_f1']:.3f}")
        var_lines.append(f"  Constant-baseline:    {var_res['avg_const_mean_error_m']:.4f} m "
                         f"(mean training position; beat this to show a transferable mapping)")
    return "\n".join(var_lines)
