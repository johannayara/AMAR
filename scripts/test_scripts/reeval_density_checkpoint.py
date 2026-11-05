"""
[file]          reeval_density_checkpoint.py
[description]   Re-score a saved density-map checkpoint offline, without retraining, under the peak
                rules either side of the position-metric fix.

                A run saves model.pth (weights + x_shape + grid/sigma/decoder config + output_mode) but
                neither the predicted maps nor the per-repeat metrics, so a metric that changes after
                the run can only be recovered by reloading the checkpoint, re-running inference on the
                held-out room, and re-applying the decision. This script does exactly that and reports
                the two rules side by side:

                  * "old" -- the pre-fix behaviour: position_error_m / position_miss_rate /
                    position_false_positive_rate scored on EVERY local maximum above the relative cut
                    (preset["density"]["peak_threshold"] * map_max), while the count used only the
                    peaks above the calibrated absolute threshold. That mismatch made
                    position_error_m non-increasing in the number of predicted peaks, so a diffuse map
                    scored better on it for free -- with the only warning sign (fp) absent from the
                    .txt report.
                  * "new" -- the fixed behaviour: the count and the positions come from one peak set
                    (the peaks above the calibrated absolute threshold).

                The count metrics are identical between the two rows by construction; only the three
                position metrics move. That is the point: this measures the fix, not the model.

                The validation split is reproduced exactly as the runner builds it (10% of each
                training room, RandomState(39), each frame scored with its own room's kernels), so the
                calibrated thresholds match the run's.

                Caveat: save_density_checkpoint keeps only the LAST repeat's weights, so this is a
                single-repeat estimate, not the run's ±SE aggregate.

                Usage:
                    # the sgd meeting_room run (reproduce the 0.79 m, then show the fixed number)
                    python scripts/reeval_density_checkpoint.py \
                        --checkpoint visualizations/cross_domain/empty_room_classroom/33/model.pth \
                        --env meeting_room --train_envs empty_room,classroom \
                        --reported 0.7875

                    # same checkpoint under the calibrated empty gate
                    python scripts/reeval_density_checkpoint.py ... --gate calibrated
"""
#
##

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
## scripts/ is sys.path[0] when this file is run directly, so the sibling helpers import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from configs.preset import preset
from src.models.density_map import (
    _peaks_with_values, build_kernels, calibrate_empty_threshold, density_map_maxima,
    evaluate_count_rule, layout_free_calibration, normalized_to_meters, occupancy_to_positions,
    position_error_metrics, resolve_count_decision, sample_location_occupancy,
)
from src.utils import select_device
from visualize_density_checkpoint import load_model, load_room, predict_density

#
## the position metrics the peak rule changes; the count metrics must not move
POSITION_KEYS = ("position_error_m", "position_miss_rate", "position_false_positive_rate")
METRIC_ORDER = (("accuracy", "exact", "%.4f"), ("balanced_accuracy", "balanced", "%.4f"),
                ("mae", "MAE", "%.4f"), ("occupancy_accuracy", "occAcc", "%.4f"),
                ("occupancy_f1", "occF1", "%.4f"), ("mass_accuracy", "massAcc", "%.4f"),
                ("mass_mae", "massMAE", "%.4f"), ("position_error_m", "posErr", "%.4f"),
                ("position_miss_rate", "miss", "%.4f"),
                ("position_false_positive_rate", "fp", "%.4f"))


def parse_args():
    var_args = argparse.ArgumentParser()
    var_args.add_argument("--checkpoint", required=True, type=str, help="path of a model.pth from a density run")
    var_args.add_argument("--env", required=True, type=str, help="the held-out/test room to score")
    var_args.add_argument("--train_envs", required=True, type=str,
                          help="comma-separated training rooms, used to rebuild the run's validation "
                               "split and its calibrated thresholds")
    var_args.add_argument("--users", default="0, 1,2,3,4,5", type=str)
    var_args.add_argument("--backbone", default=None, type=str,
                          help="override nn.backbone. Default: the value stored in the checkpoint")
    var_args.add_argument("--encoder-norm", default=None, type=str,
                          help="override nn.encoder_norm (batch|group|layer)")
    var_args.add_argument("--gate", default="off", type=str,
                          help="empty gate: 'off' (no gate, what every pre-2026-10-27 run used), "
                               "'calibrated' (the validation-calibrated gate), or a float value")
    var_args.add_argument("--reported", default=None, type=float,
                          help="the run's reported 'Position Error (m)', to check this reproduction")
    var_args.add_argument("--out", default=None, type=str, help="optional path to dump the rows as JSON")
    return var_args.parse_args()


def parse_gate(var_gate, var_calibrated):
    """
    [description]
    : resolve the --gate argument into the value evaluate_count_rule takes (None disables the gate). A
      calibrated value that is not finite is unusable, so it falls back to no gate.
    """
    var_gate = str(var_gate).strip().lower()
    if var_gate in ("off", "none", "false"):
        return None
    if var_gate in ("calibrated", "calib", "true"):
        if var_calibrated is None or not np.isfinite(var_calibrated):
            return None
        return float(var_calibrated)
    return float(var_gate)


def build_validation_split(var_model, var_train_envs, var_grid_size, var_sigma, var_device, var_users):
    """
    [description]
    : reproduce the runner's validation split: 10% of each training room under RandomState(39), each
      frame scored with its own room's kernels (the runner's var_kernel_bank[room_index]).
    : return: (valid_density (N,H,W), valid_occupancy (N,L), valid_count (N,), valid_map_max (N,))
    """
    var_density_all, var_occ_all, var_count_all, var_max_all = [], [], [], []
    for var_env in var_train_envs:
        var_x, var_y = load_room(var_env, var_users)
        var_perm = np.random.RandomState(39).permutation(len(var_x))
        var_num_valid = max(1, int(round(0.1 * len(var_x))))
        var_idx = np.sort(var_perm[:var_num_valid])
        var_kernels = torch.from_numpy(build_kernels(preset["layouts"][var_env], var_grid_size, var_sigma)[0])
        var_density = predict_density(var_model, var_x[var_idx], var_device)
        var_density_all.append(var_density.numpy())
        var_occ_all.append(sample_location_occupancy(var_density, var_kernels).numpy())
        var_count_all.append(var_y[var_idx].sum(axis=1).round())
        var_max_all.append(density_map_maxima(var_density))
    return (np.concatenate(var_density_all), np.concatenate(var_occ_all),
            np.concatenate(var_count_all), np.concatenate(var_max_all))


def relative_cut_positions(var_density, var_empty_threshold):
    """
    [description]
    : the PRE-FIX position set: every local maximum above the relative cut, the empty gate applied
      afterwards exactly as layout_free_counts did. Kept here (not in the library) because the library
      no longer produces it -- this is the behaviour being measured against.
    : return: list of (M, 2) arrays in meters, one per frame
    """
    var_frac = preset["density"]["peak_threshold"]
    var_positions = [normalized_to_meters(_peaks_with_values(var_map, var_frac, 0.0)[0])
                     for var_map in np.asarray(var_density)]
    if var_empty_threshold is not None:
        var_is_empty = density_map_maxima(var_density) < var_empty_threshold
        var_positions = [np.zeros((0, 2), dtype=np.float32) if var_empty else var_pos
                         for var_empty, var_pos in zip(var_is_empty, var_positions)]
    return var_positions


def score(var_pred_density, var_true_y, var_env, var_kernels, var_occ_threshold,
          var_empty_threshold, var_peak_threshold, var_mass, var_peak_rule="new"):
    """
    [description]
    : full layout-free metric dict of one room under either peak rule. The count metrics come from
      evaluate_count_rule in both cases (so the two rules are scored on identical counts); only the
      three position metrics are substituted when var_peak_rule is "old".
    """
    var_metrics, _ = evaluate_count_rule(
        "layout_free", var_pred_density, var_true_y, preset["layouts"][var_env], var_kernels,
        var_occ_threshold, var_empty_threshold, var_peak_threshold, var_mass)
    if var_peak_rule == "new":
        return var_metrics
    var_old_metrics = position_error_metrics(
        relative_cut_positions(var_pred_density, var_empty_threshold),
        occupancy_to_positions(var_true_y, preset["layouts"][var_env]))
    return {**var_metrics, **var_old_metrics}


def format_rows(var_rows, var_label_width=18):
    """
    [description]
    : render the metric rows as one aligned table.
    """
    var_lines = []
    var_head = (f"{'rule':<{var_label_width}}"
                + "".join(f"{var_label:>10}" for _, var_label, _ in METRIC_ORDER))
    var_lines.append(var_head)
    var_lines.append("-" * len(var_head))
    for var_name, var_metrics in var_rows:
        var_cells = ""
        for var_key, _, var_fmt in METRIC_ORDER:
            var_value = var_metrics.get(var_key)
            var_cells += f"{'nan' if var_value is None else var_fmt % var_value:>10}"
        var_lines.append(f"{var_name:<{var_label_width}}{var_cells}")
    return "\n".join(var_lines)


def run():
    var_args = parse_args()
    var_device = select_device()
    print(f"Using device: {var_device}")

    var_train_envs = [var_e.strip() for var_e in var_args.train_envs.split(",") if var_e.strip()]
    var_model, var_checkpoint = load_model(var_args.checkpoint, var_device,
                                           var_backbone=var_args.backbone,
                                           var_encoder_norm=var_args.encoder_norm)
    var_grid_size = var_checkpoint["grid_size"]
    var_sigma = var_checkpoint["sigma"]
    print(f"Checkpoint: layout '{var_checkpoint['layout_name']}', backbone "
          f"'{var_checkpoint.get('backbone')}', grid {var_grid_size} | scoring room: {var_args.env}")

    #
    ## ---- the run's decision, rebuilt from the same validation split ------------------------------
    var_valid_density, var_valid_occ, var_valid_count, var_valid_max = build_validation_split(
        var_model, var_train_envs, var_grid_size, var_sigma, var_device, var_args.users)
    var_calibrated_empty, var_occ_threshold = resolve_count_decision(
        var_valid_occ, var_valid_count, var_valid_max)
    if var_calibrated_empty is None:
        ## the preset ships density.staged_count=False, so resolve_count_decision reports "gate
        ## disabled" rather than "what the gate would be"; --gate calibrated needs the latter
        var_calibrated_empty = float(calibrate_empty_threshold(var_valid_max, var_valid_count)[0])
    var_peak_threshold, var_mass = layout_free_calibration(var_valid_density, var_valid_count)
    var_empty_threshold = parse_gate(var_args.gate, var_calibrated_empty)
    var_calibrated_log = ("nan (unusable: no empty frames in the split)"
                          if not np.isfinite(var_calibrated_empty) else f"{var_calibrated_empty:.4f}")
    print(f"Decision: Thr {var_occ_threshold:.4f} - EmptyThr "
          f"{'nan' if var_empty_threshold is None else '%.4f' % var_empty_threshold}"
          f" (calibrated {var_calibrated_log})"
          f" - peakThr {var_peak_threshold:.4f} - mass/person {var_mass:.3f}")

    #
    ## ---- test room -------------------------------------------------------------------------------
    var_x, var_y = load_room(var_args.env, var_args.users)
    var_kernels = torch.from_numpy(build_kernels(preset["layouts"][var_args.env],
                                                 var_grid_size, var_sigma)[0])
    var_pred_density = predict_density(var_model, var_x, var_device).numpy()
    print(f"Scored {len(var_x)} frames of {var_args.env}")

    var_rows = [("old", score(var_pred_density, var_y, var_args.env, var_kernels, var_occ_threshold,
                              var_empty_threshold, var_peak_threshold, var_mass, "old")),
                ("new", score(var_pred_density, var_y, var_args.env, var_kernels, var_occ_threshold,
                              var_empty_threshold, var_peak_threshold, var_mass, "new"))]
    print()
    print(format_rows(var_rows))

    var_old_err = var_rows[0][1]["position_error_m"]
    var_new_err = var_rows[1][1]["position_error_m"]
    print(f"\nposition_error_m: old {var_old_err:.4f} -> new {var_new_err:.4f} "
          f"(fp {var_rows[0][1]['position_false_positive_rate']:.3f} -> "
          f"{var_rows[1][1]['position_false_positive_rate']:.3f})")
    if var_args.reported is not None:
        print(f"run reported {var_args.reported:.4f}; this last-repeat 'old' row is "
              f"{var_old_err:.4f} (aggregate vs last repeat differ, so expect a small gap)")

    if var_args.out:
        os.makedirs(os.path.dirname(os.path.abspath(var_args.out)), exist_ok=True)
        with open(var_args.out, "w") as var_file:
            json.dump({"checkpoint": var_args.checkpoint, "env": var_args.env,
                       "train_envs": var_train_envs, "gate": var_args.gate,
                       "occ_threshold": var_occ_threshold,
                       "calibrated_empty_threshold": var_calibrated_empty,
                       "empty_threshold": var_empty_threshold,
                       "peak_threshold": var_peak_threshold, "per_person_mass": var_mass,
                       "rows": {var_name: var_metrics for var_name, var_metrics in var_rows}},
                      var_file, indent=1)
        print(f"Wrote {var_args.out}")


if __name__ == "__main__":
    run()
