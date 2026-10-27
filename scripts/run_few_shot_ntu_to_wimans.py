"""
[file]          run_few_shot_ntu_to_wimans.py
[description]   Few-shot knowledge distillation for the NTU-Fi HAR -> WiMANS activity transfer.

                Teacher : AMAR_WO_RVQ trained on every NTU-Fi HAR capture of the shared activities,
                          then frozen.
                Student : AMAR_WO_RVQ warm-started from the teacher and fine-tuned on a small fraction
                          of the WiMANS pool (--few_shot_ratio) with HungarianMatchingLoss plus
                          --kd_weight * SetDistillationLoss against the frozen teacher.
                Test    : the WiMANS query frames, held out of both the support set and the student's
                          validation slice, so the reported metrics are on frames neither model was
                          tuned on. The teacher (the source-only model, which is what
                          scripts/run_ntu_to_wimans.py reports) and the trivial majority-activity
                          baseline are reported alongside.

                --activities picks the label space and therefore what the WiMANS test set contains:

                    shared        (default) only the activity names both datasets have. NTU-Fi HAR and
                                  WiMANS share {walk} alone, so this is walk against no-object and
                                  every query frame is positive: the teacher scores perfectly and the
                                  run only shows whether fine-tuning degrades it.
                    walk_nothing  that vocabulary plus WiMANS's idle activity "nothing", so the target
                                  test is walking against nothing. The source has no "nothing"
                                  captures, so only the few-shot student can learn that class.
                    union         every activity of both datasets, mapped by name (14 classes: the 6
                                  NTU-Fi HAR classes and the 9 WiMANS activities, "walk" shared). The
                                  teacher is trained on NTU-Fi HAR's 6 and tested on all 9 WiMANS
                                  activities, the 8 target-only ones included.

                Both runners build the task through
                src.data.activity_transfer.prepare_transfer_task with the same seed, --few_shot_ratio
                and --activities, so the pure-split and few-shot numbers are on exactly the same query
                frames.

                Example:
                    python scripts/run_few_shot_ntu_to_wimans.py --activities walk_nothing --kd_weight 1.0
                    python scripts/run_few_shot_ntu_to_wimans.py --activities union --few_shot_ratio 0.1
                    python scripts/run_few_shot_ntu_to_wimans.py --activities union --kd_weight 0

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
from src.models import run_few_shot_cross_domain
from src.utils import save_run_outputs, build_run_stem, join_envs, run_timestamp, reserve_viz_dir

#
##
VAR_METRIC_NAMES = ("accuracy", "precision", "recall", "f1_score",
                    "perfect_prediction_percentage", "total_error")


def parse_args():
    """
    [description]
    : command line arguments for the NTU-Fi HAR -> WiMANS few-shot run.
    """
    #
    ##
    var_args = argparse.ArgumentParser()

    var_args.add_argument("--activities", default="shared", choices=ACTIVITY_VOCABULARIES,
                          help="label space of the transfer, and therefore what the WiMANS test set "
                               "holds: 'shared' (default) only the activity names both datasets have "
                               "(on NTU-Fi HAR x WiMANS that is {walk} alone, a walk-vs-no-object task "
                               "with no negatives); 'walk_nothing' adds WiMANS's idle activity, so the "
                               "target test is walking against nothing; 'union' keeps every activity of "
                               "both datasets (14 classes).")
    var_args.add_argument("--wimans_rooms", default=None, type=str,
                          help="comma-separated WiMANS rooms forming the support/query pool "
                               "(default preset['data']['environment'])")
    var_args.add_argument("--standardize", default="window", choices=("none", "window"),
                          help="'window' z-scores each CSI window, removing the per-device gain/AGC "
                               "scale that differs between NTU-Fi HAR and WiMANS; 'none' feeds raw "
                               "amplitude")
    var_args.add_argument("--few_shot_ratio", default=0.05, type=float,
                          help="fraction of the WiMANS pool used to fine-tune the student; it also "
                               "defines the query frames the pure-split runner is scored on")
    var_args.add_argument("--kd_weight", default=1.0, type=float,
                          help="weight of the distillation term (0 = supervised few-shot only)")
    var_args.add_argument("--kd_temperature", default=1.0, type=float,
                          help="temperature of the distillation soft targets")
    var_args.add_argument("--no_init_from_teacher", action="store_true",
                          help="start the student from random weights instead of the teacher's")
    var_args.add_argument("--no_compile", action="store_true",
                          help="disable torch.compile of the feature extractors")
    var_args.add_argument("--teacher_epochs", default=None, type=int,
                          help="override the teacher's epochs (default preset['nn']['epoch'])")
    var_args.add_argument("--student_epochs", default=None, type=int,
                          help="override the student's epochs (default preset['nn']['epoch'])")
    var_args.add_argument("--repeat", default=3, type=int, help="number of repeated experiments")
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



def format_result(var_result, var_shared, var_majority_name):
    """
    [description]
    : human-readable report: student, teacher and majority-activity baseline per target room.
    """
    #
    ##
    var_lines = ["=" * 80,
                 f"NTU-Fi HAR -> WiMANS FEW-SHOT ACTIVITY DISTILLATION "
                 f"(support {var_result['few_shot_ratio']:.2%}, kd_weight {var_result['kd_weight']}, "
                 f"kd_temperature {var_result['kd_temperature']})",
                 f"Shared activities ({var_result['vocabulary']}): {var_shared} | support frames: "
                 f"{var_result['support_frames']} | query frames: {var_result['query_frames']}",
                 "=" * 80]
    for var_room, var_env_result in var_result["per_env"].items():
        var_lines.append(f"\n{var_room}:")
        for var_who, var_note in (("student", ""), ("teacher", "  (NTU-Fi HAR teacher, no WiMANS fine-tuning)")):
            var_lines.append(f"  {var_who.capitalize():8s}" + " | ".join(
                f"{var_label} {var_env_result[f'avg_{var_who}_{var_name}']:.4f} ± "
                f"{var_env_result[f'se_{var_who}_{var_name}']:.4f} (SE)"
                for var_name, var_label in (("accuracy", "Accuracy"), ("f1_score", "F1"),
                                            ("perfect_prediction_percentage", "PPP"),
                                            ("total_error", "Total Error"))) + var_note)
        var_lines.append("  Majority" + " | ".join(
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
    : run the NTU-Fi HAR -> WiMANS few-shot distillation.
    """
    #
    ##
    SEED = 103
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    #
    var_args = parse_args()
    var_hp_applied = apply_hp_overrides(var_args.hp)
    #
    ## ------------------------------- load the shared-activity task -------------------------------
    var_task = prepare_transfer_task(
        var_wimans_rooms=([var_room.strip() for var_room in var_args.wimans_rooms.split(',')]
                          if var_args.wimans_rooms else None),
        var_standardize=var_args.standardize,
        var_few_shot_ratio=var_args.few_shot_ratio,
        var_vocabulary=var_args.activities)
    var_shared = var_task["var_shared"]
    preset["nn"]["num_classes"] = var_task["var_num_classes"]
    preset["task"] = "activity"
    if len(var_shared) == 1:
        print(f"\nNOTE: the shared vocabulary is the single activity {var_shared[0]!r}, so this is a "
              f"{var_shared[0]!r}-vs-no-object detection task and every query frame is positive: the "
              f"teacher is already perfect on it and the run only shows whether fine-tuning degrades "
              f"it. Pass --activities walk_nothing to test it against WiMANS's idle frames, or "
              f"--activities union for every activity of both datasets.")
    #
    ## --------------------------- the trivial always-majority baseline -----------------------------
    var_majority = majority_activity(var_task["ntu_class_indices"])
    var_majority_name = var_shared[var_majority] if var_majority >= 0 else None
    var_baseline = {var_room: {f"majority_{var_name}": float(majority_query_metrics(
        var_majority, var_query_classes, var_task["var_num_classes"])[var_name])
        for var_name in VAR_METRIC_NAMES}
        for var_room, var_query_classes in var_task["query_class_indices_by_env"].items()}
    #
    ## ------------------------------------ run the distillation ------------------------------------
    var_save_path = reserve_viz_dir(path="few_shot_ntu_to_wimans/ntu_har")
    var_per_env = run_few_shot_cross_domain(
        var_task["train_sets_by_env"], var_task["pool_by_env"],
        var_task["support_idx"], var_task["valid_idx"], var_task["query_idx"],
        var_few_shot_ratio=var_args.few_shot_ratio, var_kd_weight=var_args.kd_weight,
        var_kd_temperature=var_args.kd_temperature,
        var_teacher_epochs=var_args.teacher_epochs, var_student_epochs=var_args.student_epochs,
        var_init_from_teacher=not var_args.no_init_from_teacher,
        var_compile=not var_args.no_compile, var_repeat=var_args.repeat, var_task="activity",
        var_env="ntu_har", save_path=var_save_path)
    for var_room, var_metrics in var_baseline.items():
        if var_room in var_per_env:
            var_per_env[var_room].update(var_metrics)
    #
    var_result = {
        "protocol": "train NTU-Fi HAR teacher -> few-shot KD on the WiMANS pool -> test held-out "
                    "WiMANS query frames",
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
        "kd_weight": var_args.kd_weight,
        "kd_temperature": var_args.kd_temperature,
        "init_from_teacher": not var_args.no_init_from_teacher,
        "compile": not var_args.no_compile,
        "support_frames": int(len(var_task["support_idx"])),
        "valid_frames": int(len(var_task["valid_idx"])),
        "query_frames": int(len(var_task["query_idx"])),
        "repeat": var_args.repeat,
        "teacher_epochs": (var_args.teacher_epochs if var_args.teacher_epochs is not None
                           else preset["nn"]["epoch"]),
        "student_epochs": (var_args.student_epochs if var_args.student_epochs is not None
                           else preset["nn"]["epoch"]),
        "nn": preset["nn"],
        "hp_overrides": var_hp_applied,
        "per_env": var_per_env,
    }
    #
    var_formatted = format_result(var_result, var_shared, var_majority_name)
    var_stem = build_run_stem("few_shot_ntu_to_wimans", "activity",
                              f"vocab-{var_task['var_vocabulary']}",
                              f"test-{join_envs(list(var_task['query_by_env']))}",
                              f"fs{var_args.few_shot_ratio}", f"kd{var_args.kd_weight}",
                              f"r{var_args.repeat}", var_args.standardize, run_timestamp())
    var_json_path, var_txt_path = save_run_outputs(
        preset["path"].get("save_dir", "output"), "few_shot_ntu_to_wimans", var_stem,
        var_result, var_formatted)
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
