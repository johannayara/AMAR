"""
[file]          sweep_gate_checkpoint.py
[description]   Does the stage-1 empty gate do anything? Sweep it on one saved checkpoint, offline.

                The gate is the only part of the count decision that changes the reported metric in a
                way the occupancy threshold cannot: a frame whose density-map maximum is below the gate
                is predicted empty regardless of its per-location readouts, so it takes count 0 and
                drops its predicted positions. Every run committed up to 2026-10-27 had it disabled
                (preset density.staged_count was None, read as falsy), which is visible in those logs as
                "EmptyThr nan" and in their per-count tables as an accuracy near 0 on the true-empty
                class.

                This script reloads a checkpoint, rebuilds the run's validation split, runs inference on
                the held-out room once, and scores the SAME predicted maps under a list of gate values:

                  off          -- no gate (the current committed behaviour)
                  calibrated   -- the gate the validation split picks (what a gate-on run would use)
                  0.01 .. 0.30 -- explicit map-maximum thresholds, to see the sensitivity

                The count metrics all move with the gate; the model does not. So the row for "off" must
                reproduce the run's reported numbers (up to the last-repeat-vs-aggregate difference) and
                the other rows price the gate.

                Caveat: save_density_checkpoint keeps only the LAST repeat's weights, so this is a
                single-repeat estimate, not the run's ±SE aggregate.

                Usage:
                    python scripts/sweep_gate_checkpoint.py \
                        --checkpoint visualizations/cross_domain/empty_room_classroom/33/model.pth \
                        --env meeting_room --train_envs empty_room,classroom

                    python scripts/sweep_gate_checkpoint.py ... --gates off,calibrated,0.05,0.10
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
    build_kernels, density_map_maxima, layout_free_calibration, resolve_count_decision,
)
from src.utils import select_device
from reeval_density_checkpoint import (
    build_validation_split, format_rows, load_model, load_room, parse_gate, predict_density, score,
)

#
## the columns this sweep adds on top of the shared metric order
DEFAULT_GATES = "off,calibrated,0.01,0.02,0.04,0.06,0.08,0.10,0.15,0.20,0.30"


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
                          help=f"comma-separated gate values; 'off' disables the gate and 'calibrated' "
                               f"uses the validation-calibrated one (default: {DEFAULT_GATES})")
    var_args.add_argument("--peak_rule", default="new", choices=("new", "old"),
                          help="which peak set the position metrics use; 'new' is the fixed one")
    var_args.add_argument("--out", default=None, type=str, help="optional path to dump the rows as JSON")
    return var_args.parse_args()


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
    var_calibrated_empty, var_occ_threshold = resolve_count_decision(
        var_valid_occ, var_valid_count, var_valid_max)
    var_peak_threshold, var_mass = layout_free_calibration(var_valid_density, var_valid_count)
    print(f"Fixed part of the decision: Thr {var_occ_threshold:.4f} - peakThr "
          f"{var_peak_threshold:.4f} - mass/person {var_mass:.3f} | calibrated gate "
          f"{'nan (degenerate split)' if var_calibrated_empty is None else '%.4f' % var_calibrated_empty}")

    var_x, var_y = load_room(var_args.env, var_args.users)
    var_kernels = torch.from_numpy(build_kernels(preset["layouts"][var_args.env],
                                                 var_grid_size, var_sigma)[0])
    var_pred_density = predict_density(var_model, var_x, var_device).numpy()
    var_map_max = density_map_maxima(var_pred_density)
    print(f"Scored {len(var_x)} frames of {var_args.env}\n")

    var_rows, var_records = [], {}
    for var_token in [var_g.strip() for var_g in var_args.gates.split(",") if var_g.strip()]:
        var_gate = parse_gate(var_token, var_calibrated_empty)
        var_metrics = score(var_pred_density, var_y, var_args.env, var_kernels, var_occ_threshold,
                            var_gate, var_peak_threshold, var_mass, var_args.peak_rule)
        var_rows.append((var_token, var_metrics))
        var_records[var_token] = {**var_metrics, "gate_value": var_gate,
                                  "gated_fraction": (float(np.mean(var_map_max < var_gate))
                                                     if var_gate is not None else 0.0)}
    print(format_rows(var_rows))

    ## per-count accuracy on the true-empty frames is the metric the gate exists to move, so print it
    ## separately rather than letting it hide inside the exact-count average
    print(f"\n{'gate':<12}{'gated%':>8}{'acc@0':>8}{'acc@1':>8}{'acc@2':>8}{'MAE':>8}{'fp':>8}")
    for var_token, var_metrics in var_rows:
        var_record = var_records[var_token]
        var_per_class = var_metrics.get("per_class_accuracy", {})
        print(f"{var_token:<12}{100 * var_record['gated_fraction']:>7.1f}%"
              + "".join(f"{var_per_class.get(var_class, float('nan')):>8.3f}" for var_class in (0, 1, 2))
              + f"{var_metrics['mae']:>8.3f}{var_metrics['position_false_positive_rate']:>8.3f}")

    if var_args.out:
        os.makedirs(os.path.dirname(os.path.abspath(var_args.out)), exist_ok=True)
        with open(var_args.out, "w") as var_file:
            json.dump({"checkpoint": var_args.checkpoint, "env": var_args.env,
                       "train_envs": var_train_envs, "peak_rule": var_args.peak_rule,
                       "occ_threshold": var_occ_threshold,
                       "calibrated_empty_threshold": var_calibrated_empty,
                       "peak_threshold": var_peak_threshold, "per_person_mass": var_mass,
                       "rows": var_records}, var_file, indent=1)
        print(f"\nWrote {var_args.out}")


if __name__ == "__main__":
    run()
