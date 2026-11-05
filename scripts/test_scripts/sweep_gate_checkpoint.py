"""
[file]          sweep_gate_checkpoint.py
[description]   Does the stage-1 empty gate do anything? Sweep it on one saved checkpoint, offline.

                The gate is the only part of the count decision that changes the reported metric in a
                way the occupancy threshold cannot: a frame whose density-map maximum is below the gate
                is predicted empty regardless of its per-location readouts, so it takes count 0 and
                drops its predicted positions. Every run committed up to 2026-10-27 had it disabled
                (preset density.staged_count was None, read as falsy), which shows in those logs as
                "EmptyThr nan" and in their per-count tables as an accuracy near 0 on the true-empty
                class.

                This script reloads a checkpoint, rebuilds the run's validation split, runs inference on
                the held-out room once, and scores the SAME predicted maps under a list of gate values.

                TWO THINGS MAKE A GATE VALUE MEANINGLESS, and both are reported rather than hidden:

                1. A gate at or below the calibrated peak threshold cannot change the count. A peak
                   counts when its value clears var_peak_threshold, and every peak value is at most the
                   map maximum, so a frame with map max < peak_threshold already has a peak count of 0
                   -- gating it to 0 is a no-op. The gate only moves the count once gate > peakThr.
                   Only the mass count (round(map_sum / per_person_mass), which is not peak-based) can
                   move below that. That is why the default grid is anchored on peakThr and on
                   percentiles of the observed map maxima instead of a fixed 0.01..0.30 range.
                2. The calibrated gate can be degenerate. calibrate_empty_threshold returns -inf when
                   the validation split has no empty frames and +inf when all of them are, which means
                   the split cannot place the gate at all; that is printed, and the "calibrated" row
                   then falls back to no gate.

                The gate is calibrated here directly (calibrate_empty_threshold) rather than through
                resolve_count_decision, because the preset ships density.staged_count=False: going
                through resolve_count_decision would return "gate disabled" for every run and make the
                "calibrated" row a duplicate of "off".

                The count metrics all move with the gate; the model does not. So the row for "off" must
                reproduce the run's reported numbers (up to the last-repeat-vs-aggregate difference) and
                the other rows price the gate.

                Caveat: save_density_checkpoint keeps only the LAST repeat's weights, so this is a
                single-repeat estimate, not the run's ±SE aggregate.

                Usage:
                    python scripts/sweep_gate_checkpoint.py \
                        --checkpoint visualizations/cross_domain/empty_room_classroom/33/model.pth \
                        --env meeting_room --train_envs empty_room,classroom

                    python scripts/sweep_gate_checkpoint.py ... --gates off,calibrated,0.60,0.70,0.80
"""
#
##

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from configs.preset import preset
from src.models.density_map import (
    build_kernels, calibrate_empty_threshold, density_map_maxima, layout_free_calibration,
    resolve_count_decision,
)
from src.utils import select_device
from reeval_density_checkpoint import (
    build_validation_split, format_rows, load_model, load_room, predict_density, score,
)

#
## anchored on the effective range: off, the calibrated value, the structural boundary (peakThr), and
## percentiles of the validation map maxima -- a fixed 0.01..0.30 grid sits entirely below peakThr and
## can only move the mass count
DEFAULT_GATES = "off,calibrated,peakThr,q05,q10,q25,q50,q75,q90"


def parse_args():
    var_args = argparse.ArgumentParser()
    var_args.add_argument("--checkpoint", required=True, type=str, help="path of a model.pth from a density run")
    var_args.add_argument("--env", required=True, type=str, help="the held-out/test room to score")
    var_args.add_argument("--train_envs", required=True, type=str,
                          help="comma-separated training rooms, used to rebuild the run's validation "
                               "split and its calibrated thresholds")
    var_args.add_argument("--users", default="0, 1,2,3,4,5", type=str)
    var_args.add_argument("--backbone", default=None, type=str)
    var_args.add_argument("--encoder-norm", default=None, type=str)
    var_args.add_argument("--gates", default=DEFAULT_GATES, type=str,
                          help="comma-separated gate tokens: 'off', 'calibrated', 'peakThr', 'qNN' "
                               "(NN-th percentile of the validation map maxima), or a float "
                               f"(default: {DEFAULT_GATES})")
    var_args.add_argument("--peak_rule", default="new", choices=("new", "old"),
                          help="which peak set the position metrics use; 'new' is the fixed one")
    var_args.add_argument("--out", default=None, type=str, help="optional path to dump the rows as JSON")
    return var_args.parse_args()


def parse_gates(var_spec, var_valid_max, var_peak_threshold, var_calibrated):
    """
    [description]
    : expand the --gates spec into [(row label, gate value)] with None meaning "no gate".
    """
    var_rows = []
    for var_token in [var_t.strip() for var_t in var_spec.split(",") if var_t.strip()]:
        var_lower = var_token.lower()
        if var_lower in ("off", "none"):
            var_rows.append(("off", None))
        elif var_lower == "calibrated":
            var_rows.append(("calibrated", var_calibrated))
        elif var_lower == "peakthr":
            var_rows.append(("peakThr", float(var_peak_threshold)))
        elif var_lower.startswith("q") and var_lower[1:].replace(".", "", 1).isdigit():
            var_rows.append((var_token, float(np.percentile(var_valid_max, float(var_token[1:])))))
        else:
            var_rows.append((var_token, float(var_token)))
    return [(var_label if var_value is None else f"{var_label}={var_value:.4f}", var_value)
            for var_label, var_value in var_rows]


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

    var_valid_density, var_valid_occ, var_valid_count, var_valid_max = build_validation_split(
        var_model, var_train_envs, var_grid_size, var_sigma, var_device, var_args.users)
    var_occ_threshold = resolve_count_decision(var_valid_occ, var_valid_count, var_valid_max)[1]
    var_peak_threshold, var_mass = layout_free_calibration(var_valid_density, var_valid_count)
    ## calibrate the gate directly: resolve_count_decision returns "disabled" while the preset ships
    ## density.staged_count=False, which would make every "calibrated" row a copy of "off"
    var_calibrated, var_calibrated_score = calibrate_empty_threshold(var_valid_max, var_valid_count)
    var_calibrated = float(var_calibrated) if np.isfinite(var_calibrated) else None
    print(f"Fixed part of the decision: Thr {var_occ_threshold:.4f} - peakThr {var_peak_threshold:.4f}"
          f" - mass/person {var_mass:.3f}")
    print(f"Validation map maxima: min {var_valid_max.min():.4f} - p10 "
          f"{np.percentile(var_valid_max, 10):.4f} - p50 {np.percentile(var_valid_max, 50):.4f} - "
          f"p90 {np.percentile(var_valid_max, 90):.4f} - max {var_valid_max.max():.4f}")
    if var_calibrated is None:
        print(f"Calibrated gate: NOT USABLE (balanced empty/non-empty acc "
              f"{var_calibrated_score:.3f}) -- the validation split has no empty frames or is all "
              f"empty, so the 'calibrated' row falls back to no gate")
    else:
        print(f"Calibrated gate: {var_calibrated:.4f} (balanced empty/non-empty acc "
              f"{var_calibrated_score:.3f})")

    var_x, var_y = load_room(var_args.env, var_args.users)
    var_kernels = torch.from_numpy(build_kernels(preset["layouts"][var_args.env],
                                                 var_grid_size, var_sigma)[0])
    var_pred_density = predict_density(var_model, var_x, var_device).numpy()
    var_map_max = density_map_maxima(var_pred_density)
    print(f"Scored {len(var_x)} frames of {var_args.env}: map maxima min {var_map_max.min():.4f} - "
          f"p50 {np.percentile(var_map_max, 50):.4f} - max {var_map_max.max():.4f}\n")

    var_rows, var_records = [], {}
    for var_label, var_gate in parse_gates(var_args.gates, var_valid_max, var_peak_threshold,
                                           var_calibrated):
        var_metrics = score(var_pred_density, var_y, var_args.env, var_kernels, var_occ_threshold,
                            var_gate, var_peak_threshold, var_mass, var_args.peak_rule)
        var_rows.append((var_label, var_metrics))
        var_records[var_label] = {**var_metrics, "gate_value": var_gate,
                                  "gated_fraction": (float(np.mean(var_map_max < var_gate))
                                                     if var_gate is not None else 0.0)}
    print(format_rows(var_rows))

    ## per-count accuracy on the true-empty frames is the metric the gate exists to move, so print it
    ## separately rather than letting it hide inside the exact-count average
    print(f"\n{'gate':<18}{'gated%':>8}{'acc@0':>8}{'acc@1':>8}{'acc@2':>8}{'MAE':>8}{'fp':>8}")
    for var_label, var_metrics in var_rows:
        var_record = var_records[var_label]
        var_per_class = var_metrics.get("per_class_accuracy", {})
        print(f"{var_label:<18}{100 * var_record['gated_fraction']:>7.1f}%"
              + "".join(f"{var_per_class.get(var_class, float('nan')):>8.3f}" for var_class in (0, 1, 2))
              + f"{var_metrics['mae']:>8.3f}{var_metrics['position_false_positive_rate']:>8.3f}")
    if all(var_records[var_label]["gated_fraction"] == 0.0 for var_label, _ in var_rows):
        print("\nno gate value fired: every frame's map maximum is above every threshold swept")

    if var_args.out:
        os.makedirs(os.path.dirname(os.path.abspath(var_args.out)), exist_ok=True)
        with open(var_args.out, "w") as var_file:
            json.dump({"checkpoint": var_args.checkpoint, "env": var_args.env,
                       "train_envs": var_train_envs, "peak_rule": var_args.peak_rule,
                       "occ_threshold": var_occ_threshold,
                       "calibrated_empty_threshold": var_calibrated,
                       "calibrated_empty_score": var_calibrated_score,
                       "peak_threshold": var_peak_threshold, "per_person_mass": var_mass,
                       "valid_map_max": var_valid_max.tolist(),
                       "test_map_max": var_map_max.tolist(),
                       "rows": var_records}, var_file, indent=1)
        print(f"\nWrote {var_args.out}")


if __name__ == "__main__":
    run()
