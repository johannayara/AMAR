"""
[file]          run_ntu_to_wimans.py
[description]   Pure cross-dataset split: train AMAR_WO_RVQ on the NTU-Fi HAR activities that WiMANS
                also labels, then test it on the WiMANS version of those activities with no WiMANS
                training at all.

                The two datasets label different vocabularies (NTU-Fi HAR: box, circle, clean, fall,
                run, walk; WiMANS: nothing, walk, rotation, jump, wave, lie_down, pick_up, sit_down,
                stand_up), so a cross-dataset activity run is only well defined on the activities both
                of them have. --activities picks the label space -- "shared" (only the names both use),
                "walk_nothing" (walk against WiMANS's idle frames) or "union" (every activity of both)
                -- and maps it onto one class ordering plus the AMAR_WO_RVQ no-object class, so the
                model's output space is identical on the source and the target.

                This runner and scripts/run_few_shot_ntu_to_wimans.py both build the task through
                src.data.activity_transfer.prepare_transfer_task, which splits the WiMANS pool with one
                fixed seed. Both are therefore scored on exactly the same query frames, so the
                pure-split and few-shot numbers are directly comparable.

                Every report carries the trivial "always name the most frequent NTU-Fi HAR activity"
                baseline. On a shared vocabulary of a single activity a model that always names it is
                already perfect, so a run that only matches this baseline has learned nothing.

                Example:
                    python scripts/run_ntu_to_wimans.py --repeat 3 --epochs 60
                    python scripts/run_ntu_to_wimans.py --activities union --repeat 3 --epochs 60

                Dataset paths, features, band, window length and occupant filter come from the preset
                (see configs/preset.py); --hp overrides any of them for one run, e.g.
                --hp ntu_har.path=/data/ntu_har.
"""
#
##

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.preset import preset
from src.data.activity_transfer import (ACTIVITY_VOCABULARIES, majority_activity,
                                        majority_query_metrics, prepare_transfer_task)
from src.models import run_cross_domain
from src.utils import save_run_outputs, build_run_stem, join_envs, run_timestamp, reserve_viz_dir

#
##
VAR_METRIC_NAMES = ("accuracy", "precision", "recall", "f1_score",
                    "perfect_prediction_percentage", "total_error")


def parse_args():
    """
    [description]
    : command line arguments for the NTU-Fi HAR -> WiMANS pure-split run.
    """
    #
    ##
    var_args = argparse.ArgumentParser()
    var_args.add_argument("--activities", default="shared", choices=ACTIVITY_VOCABULARIES,
                          help="label space of the transfer: 'shared' (default) keeps only the activity "
                               "names both datasets have (on NTU-Fi HAR x WiMANS that is {walk} alone, a "
                               "walk-vs-no-object task with no negatives); 'walk_nothing' adds WiMANS's "
                               "idle activity, so the target test is walking against nothing; 'union' "
                               "keeps every activity of both datasets (14 classes).")
    var_args.add_argument("--wimans_rooms", default=None, type=str,
                          help="comma-separated WiMANS rooms (default preset['data']['environment'])")
    var_args.add_argument("--standardize", default="window", choices=("none", "window"),
                          help="'window' z-scores each CSI window, removing the per-device gain/AGC "
                               "scale that differs between NTU-Fi HAR and WiMANS; 'none' feeds raw "
                               "amplitude")
    var_args.add_argument("--few_shot_ratio", default=0.05, type=float,
                          help="the pool split the few-shot runner uses; here it only defines the query "
                               "frames this run is scored on, so both protocols report the same frames")
    var_args.add_argument("--repeat", default=3, type=int, help="number of repeated experiments")
    var_args.add_argument("--epochs", default=None, type=int,
                          help="override preset['nn']['epoch']. The cosine LR schedule is tied to this "
                               "value, so it must match the actual training length.")
    var_args.add_argument("--hp", action="append", default=None, type=str, metavar="KEY=VALUE",
                          help="Override a preset field, dotted path, e.g. --hp nn.backbone=thatv. "
                               "Repeatable. Values are parsed as bool/int/float, else kept as str.")
    #
    return var_args.parse_args()


def parse_hp_value(var_value):
    """
    [description]
    : parse the right-hand side of a --hp override into a bool / int / float / str.
    """
    var_value = var_value.strip()
    ## accept a quoted value, e.g. --hp path.save_dir='/tmp/x'
    if len(var_value) >= 2 and var_value[0] == var_value[-1] and var_value[0] in ("'", '"'):
        return var_value[1:-1]
    var_lower = var_value.lower()
    if var_lower in ("true", "false"):
        return var_lower == "true"
    if var_lower in ("none", "null"):
        return None
    try:
        return int(var_value)
    except ValueError:
        pass
    try:
        return float(var_value)
    except ValueError:
        pass
    return var_value


def apply_hp_overrides(var_overrides):
    """
    [description]
    : apply --hp overrides onto the module-level preset dict, in place, before any model is built.
    : return: dict, the applied {dotted_key: value} for the result record
    """
    var_applied = {}
    for var_item in var_overrides or []:
        var_key, var_sep, var_value = var_item.partition("=")
        if not var_sep:
            raise ValueError(f"--hp expects KEY=VALUE, got {var_item!r}")
        var_target = preset
        var_parts = var_key.strip().split(".")
        for var_part in var_parts[:-1]:
            if var_part not in var_target:
                raise KeyError(f"--hp path {var_key!r}: preset has no {var_part!r}")
            var_target = var_target[var_part]
        var_parsed = parse_hp_value(var_value)
        var_target[var_parts[-1]] = var_parsed
        var_applied[var_key.strip()] = var_parsed
    return var_applied




def scalar_metrics(var_metrics):
    """
    [description]
    : the scalar entries of a calculate_scores() dict (it also carries per-person arrays, which do not
      belong in a JSON report), namespaced "majority_" so they sit beside the model's own
      "avg_<metric>" entries in the same result dict instead of overwriting them.
    """
    return {f"majority_{var_name}": float(var_metrics[var_name]) for var_name in VAR_METRIC_NAMES}


def format_result(var_result, var_shared, var_majority_name):
    """
    [description]
    : human-readable report: the model next to the majority-activity baseline, per target room.
    """
    #
    ##
    var_lines = ["=" * 80,
                 f"NTU-Fi HAR -> WiMANS ACTIVITY TRANSFER, PURE SPLIT (model {var_result['model']})",
                 f"Shared activities ({var_result['vocabulary']}): {var_shared} | query frames: "
                 f"{var_result['query_frames']}",
                 "=" * 80]
    for var_room, var_env_result in var_result["per_env"].items():
        var_lines.append(f"\n{var_room}:")
        ## run_cross_domain aggregates the perfect-prediction rate under "PPP", while the baseline
        ## goes through calculate_scores(), which names it "perfect_prediction_percentage"
        var_lines.append("  Model     " + " | ".join(
            f"{var_label} {var_env_result[f'avg_{var_name}']:.4f} ± "
            f"{var_env_result.get(f'se_{var_name}', 0.0):.4f} (SE)"
            for var_name, var_label in (("accuracy", "Accuracy"), ("f1_score", "F1"),
                                        ("PPP", "PPP"), ("total_error", "Total Error"))))
        var_lines.append("  Majority  " + " | ".join(
            f"{var_label} {var_env_result[f'majority_{var_name}']:.4f}"
            for var_name, var_label in (("accuracy", "Accuracy"), ("f1_score", "F1"),
                                        ("perfect_prediction_percentage", "PPP"),
                                        ("total_error", "Total Error"))))
    var_lines.append(f"\nMajority activity: {var_majority_name} (the most frequent NTU-Fi HAR "
                     f"activity; always predicting it is the reference to beat)")
    return "\n".join(var_lines)


def run():
    """
    [description]
    : run the NTU-Fi HAR -> WiMANS pure split.
    """
    #
    ##
    SEED = 103
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    #
    var_args = parse_args()
    if var_args.epochs is not None:
        preset["nn"]["epoch"] = var_args.epochs
    var_hp_applied = apply_hp_overrides(var_args.hp)
    #
    ## ------------------------------- load the shared-activity task -------------------------------
    var_task = prepare_transfer_task(
        var_wimans_rooms=([var_room.strip() for var_room in var_args.wimans_rooms.split(',')]
                          if var_args.wimans_rooms else None),
        var_standardize=var_args.standardize,
        var_few_shot_ratio=var_args.few_shot_ratio,
        var_vocabulary=var_args.activities,
        ## this runner is scored on the per-room query frames and never touches the pool they were cut
        ## from, so the pool is released instead of staying resident beside them
        var_with_pool=False)
    var_shared = var_task["var_shared"]
    preset["nn"]["num_classes"] = var_task["var_num_classes"]
    preset["task"] = "activity"
    if len(var_shared) == 1:
        print(f"\nNOTE: the shared vocabulary is the single activity {var_shared[0]!r}, so this is a "
              f"{var_shared[0]!r}-vs-no-object detection task and every query frame is positive. "
              f"Pass --activities walk_nothing to test it against WiMANS's idle frames, or "
              f"--activities union for every activity of both datasets.")
    #
    ## --------------------------- the trivial always-majority baseline -----------------------------
    var_majority = majority_activity(var_task["ntu_class_indices"])
    var_majority_name = var_shared[var_majority] if var_majority >= 0 else None
    var_baseline = {var_room: scalar_metrics(majority_query_metrics(
        var_majority, var_query_classes, var_task["var_num_classes"]))
        for var_room, var_query_classes in var_task["query_class_indices_by_env"].items()}
    #
    ## --------------------------------------- run the model ----------------------------------------
    var_save_path = reserve_viz_dir(path="ntu_to_wimans/ntu_har")
    var_per_env = run_cross_domain(var_task["train_sets_by_env"], var_task["query_by_env"],
                                   var_repeat=var_args.repeat, var_task="activity",
                                   var_env=["ntu_har"], save_path=var_save_path)
    for var_room, var_metrics in var_baseline.items():
        if var_room in var_per_env:
            var_per_env[var_room].update(var_metrics)
    #
    var_result = {
        "protocol": "train NTU-Fi HAR -> test WiMANS shared activities, no WiMANS training",
        "model": "AMAR_WO_RVQ",
        "task": "activity",
        "shared_activities": var_shared,
        "vocabulary": var_task["var_vocabulary"],
        "ntu_classes": var_task["var_ntu_classes"],
        "majority_activity": var_majority_name,
        "num_classes": var_task["var_num_classes"],
        "num_obj_queries": preset["nn"]["num_obj_queries"],
        "ntu_root": preset["ntu_har"]["path"],
        "ntu_splits": preset["ntu_har"]["splits"],
        "features": preset["ntu_har"]["features"],
        "wimans_rooms": var_args.wimans_rooms or preset["data"]["environment"],
        "wimans_users": "1",
        "band": preset["data"]["wifi_band"],
        "length": preset["data"]["length"],
        "standardize": var_args.standardize,
        "few_shot_ratio": var_args.few_shot_ratio,
        "query_frames": int(len(var_task["query_idx"])),
        "support_frames": int(len(var_task["support_idx"])),
        "repeat": var_args.repeat,
        "epochs": preset["nn"]["epoch"],
        "nn": preset["nn"],
        "hp_overrides": var_hp_applied,
        "per_env": var_per_env,
    }
    #
    var_formatted = format_result(var_result, var_shared, var_majority_name)
    var_stem = build_run_stem("ntu_to_wimans", "activity", f"vocab-{var_task['var_vocabulary']}",
                              f"test-{join_envs(list(var_task['query_by_env']))}",
                              f"r{var_args.repeat}", f"e{preset['nn']['epoch']}",
                              var_args.standardize, run_timestamp())
    var_json_path, var_txt_path = save_run_outputs(
        preset["path"].get("save_dir", "output"), "ntu_to_wimans", var_stem, var_result, var_formatted)
    #
    print(var_formatted)
    print(f"\nResults saved to: {var_json_path}")
    print(f"Report saved to:  {var_txt_path}")


if __name__ == "__main__":
    #
    ##
    var_start = time.time()
    run()
    print("Total time: %s seconds" % (time.time() - var_start))
