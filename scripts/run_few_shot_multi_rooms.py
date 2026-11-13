"""
[file]          run_few_shot_multi_rooms.py
[description]   Few-shot knowledge distillation runner for room_localizer with wimans rooms.

                The teacher is trained on the training rooms; the student is trained on support rooms or a 
                small fraction of test rooms while being distilled from the frozen teacher.
"""
#
##

import argparse
import gc
import os
import random
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.preset import preset
from src.data.hwild import room_names as hwild_room_names, load_hwild_room, HWILD_ROOMS
from src.data.unified_rooms import (load_wimans_room_targets, normalize_xy_meters,
                                    wimans_room_names)
from src.models.room_localization import run_few_shot_localization, format_few_shot_result
from src.utils import save_run_outputs, build_run_stem, join_envs, run_timestamp, reserve_viz_dir

#
##


def parse_args():
    """
    [description]
    : parse arguments from input
    """
    var_args = argparse.ArgumentParser()
    var_args.add_argument("--model", default="room_localizer", type=str)
    var_args.add_argument("--task", default="localization", type=str)
    var_args.add_argument("--repeat", default=preset["repeat"], type=int)
    var_args.add_argument("--users", default="0,1,2,3,4,5", type=str, help="Comma-separated list of user IDs")
    var_args.add_argument("--window", default=None, type=int, help="window length in packets")
    var_args.add_argument("--max_samples", default=None, type=int,
                          help="cap the WiMANS samples drawn per room (quick runs / smoke tests)")
    var_args.add_argument("--features", default="full", choices=("avg", "full"),
                          help="'full' (default) keeps the whole (time, 3 TX, 3 RX, 30) CSI grid as 270 "
                               "features; 'avg' averages one antenna axis to 90, matching the H-WILD "
                               "layout the cross-dataset (H-WILD -> WiMANS) runs need")
    var_args.add_argument("--no_compile", action="store_true", help="disable torch.compile of the feature extractors")
    var_args.add_argument("--train_rooms", default=None, type=str,
                          help="comma-separated teacher rooms (default: every WiMANS room, real and "
                               "synthetic)")
    var_args.add_argument("--test_rooms", default=None, type=str,
                          help="comma-separated rooms forming the query pool "
                               "(default: WiMANS)")
    var_args.add_argument("--support_rooms", default=None, type=str,
                          help="comma-separated rooms the few-shot support set is drawn from. "
                               "Default: the query pool itself, so the support is spread across "
                               "every --test_rooms room.")
    var_args.add_argument("--few_shot_ratio", default=0.05, type=float,
                          help="fraction of the support pool (see --support_rooms) used to "
                               "fine-tune the student")
    var_args.add_argument("--kd_weight", default=1.0, type=float,
                          help="weight of the teacher distillation term (0 = supervised few-shot only)")
    var_args.add_argument("--teacher_epochs", default=None, type=int,
                          help="override the teacher's epochs (default preset['hwild']['epoch'])")
    var_args.add_argument("--student_epochs", default=None, type=int,
                          help="override the student's epochs (default preset['hwild']['epoch'])")
    var_args.add_argument("--no_init_from_teacher", action="store_true",
                          help="start the student from random weights instead of the teacher's")
    var_args.add_argument("--standardize", default="none", choices=("none", "window"),
                          help="'window' z-scores each CSI window (removes the per-capture gain/AGC "
                               "scale that differs between the two datasets); 'none' feeds raw amplitude")
    var_args.add_argument("--hp", action="append", default=None, type=str, metavar="KEY=VALUE",
                          help="Override a preset field, dotted path, e.g. "
                               "--hp data.synth_wimans.enabled=true. Repeatable. Values are parsed as "
                               "bool/int/float, else kept as str.")
    #
    return var_args.parse_args()


def format_result(var_model, var_task, result, var_few_shot_ratio, var_kd_weight, var_env):
    """
    [description]
    : build a formatted string of the per-environment results
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"EXPERIMENT RESULTS - Model: {var_model}, Task: {var_task}")
    lines.append(f"Train env: {var_env} | Few-shot ratio: {var_few_shot_ratio} | KD weight: {var_kd_weight}")
    lines.append("=" * 80)
    lines.append("PER_ENV_RESULTS:")
    for env_name in sorted(k for k in result.keys()
                           if isinstance(result[k], dict) and "avg_accuracy" in result[k]):
        stats = result[env_name]
        lines.append(f"\n{env_name.upper()}:")
        if "avg_mae" in stats:
            ## density-map group-count metrics
            lines.append(f"  Avg Exact-count Accuracy: {stats['avg_accuracy']:.4f} "
                         f"± {stats['se_accuracy']:.4f} (SE)")
            if "avg_balanced_accuracy" in stats:
                lines.append(f"  Avg Balanced-count Accuracy: {stats['avg_balanced_accuracy']:.4f} "
                             f"± {stats['se_balanced_accuracy']:.4f} (SE)")
            lines.append(f"  Avg Count MAE: {stats['avg_mae']:.4f} ± {stats['se_mae']:.4f} (SE)")
            lines.append(f"  Avg Occupancy Accuracy: {stats['avg_occupancy_accuracy']:.4f} "
                         f"± {stats['se_occupancy_accuracy']:.4f} (SE)")
            lines.append(f"  Avg Occupancy F1: {stats['avg_occupancy_f1']:.4f} "
                         f"± {stats['se_occupancy_f1']:.4f} (SE)")
            if "avg_mass_accuracy" in stats:
                lines.append(f"  Avg Layout-free Mass Accuracy: {stats['avg_mass_accuracy']:.4f} "
                             f"± {stats['se_mass_accuracy']:.4f} (SE)")
                lines.append(f"  Avg Layout-free Mass MAE: {stats['avg_mass_mae']:.4f} "
                             f"± {stats['se_mass_mae']:.4f} (SE)")
                lines.append(f"  Avg Position Error (m): {stats['avg_position_error_m']:.4f} "
                             f"± {stats['se_position_error_m']:.4f} (SE)")
                lines.append(f"  Avg Position Miss Rate: {stats['avg_position_miss_rate']:.4f} "
                             f"± {stats['se_position_miss_rate']:.4f} (SE)")
        else:
            for metric, label in (("precision", "Precision"), ("recall", "Recall"),
                                  ("PPP", "Perfect Prediction %"), ("f1_score", "F1 Score"),
                                  ("accuracy", "Accuracy"), ("total_error", "Total Error")):
                if f"avg_{metric}" in stats:
                    lines.append(f"  Avg {label}: {stats[f'avg_{metric}']:.4f} "
                                 f"± {stats[f'se_{metric}']:.4f} (SE)")
    return "\n".join(lines)

def standardize_windows(var_x):
    """
    [description]
    : z-score every CSI window over its time x subcarrier entries, removing the per-device gain/AGC
      level so only the window's frequency-selective shape is left. The simulated WiMANS source has a
      different capture gain from the real one, so this is what makes a synthetic-room-trained model
      comparable with the real rooms at test time.
    : var_x: (N, window, features) float32
    : return: same shape, float32
    """
    var_mu = var_x.mean(axis=(1, 2), keepdims=True)
    var_sd = var_x.std(axis=(1, 2), keepdims=True) + 1e-6
    return ((var_x - var_mu) / var_sd).astype(np.float32)


def load_room_wimans(var_room, var_args, var_tag):
    """
    [description]
    : load one WiMANS room, real ("<room>") or synthetic ("<room><suffix>"), as single-target (X, XY)
      samples in the transmitter-anchored metric frame; only the single-occupant frames are kept,
      since the localizer predicts one position. load_wimans_room_targets routes a synthetic key to
      the simulated source through wimans_room_source (preset["data"]["synth_wimans"]) while keeping
      its real room's layout and physical frame. --features picks the CSI layout: 270 (full 3 x 3
      grid, the default for this WiMANS-only run) or 90 (one antenna axis averaged).
    : return: (X, XY)
    """
    #
    if var_room in wimans_room_names():
        var_x, var_y = load_wimans_room_targets(var_room, var_max_samples=var_args.max_samples,
                                                var_length=preset["hwild"]["window"],
                                                var_antennas=var_args.features)
        if len(var_x) == 0:
            raise ValueError(f"no single-occupant WiMANS frames for {var_room!r}")
        print(f"[{var_tag}]{var_room:12s} X {var_x.shape} XY {var_y.shape} "
              f"({len(np.unique(var_y.round(3), axis=0))} distinct positions)")
        return var_x, var_y
    raise ValueError(f"unknown room {var_room!r}; not in "
                     f"{wimans_room_names()}")

#
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
    : apply --hp overrides onto the module-level preset dict, in place, before any room is resolved or
      model is built.
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

##
def run():
    """
    [description]
    : run few-shot knowledge distillation for the room_localizer. The teacher is trained on the
      --train_rooms rooms (real or synthetic WiMANS), then frozen; the student is warm-started from it
      and fine-tuned on a --few_shot_ratio fraction of the --support_rooms pool (default: the query
      rooms) with the supervised Huber position loss plus --kd_weight * a smooth-L1 match to the
      teacher. Every --test_rooms room is scored on the frames held out of both the support set and
      the student's validation slice.
    """
    SEED = 103
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    #
    var_args = parse_args()
    var_hp_applied = apply_hp_overrides(var_args.hp)
    var_task = var_args.task
    var_model = var_args.model
    var_repeat = var_args.repeat
    var_users = [u.strip() for u in var_args.users.split(',')]

    var_train_rooms = ([var_room.strip() for var_room in var_args.train_rooms.split(',')]
                       if var_args.train_rooms else wimans_room_names())
    var_test_rooms = ([var_room.strip() for var_room in var_args.test_rooms.split(',')]
                      if var_args.test_rooms else wimans_room_names())
    var_support_rooms = ([var_room.strip() for var_room in var_args.support_rooms.split(',')]
                         if var_args.support_rooms else None)

    if var_support_rooms is None:
        var_real_query_rooms = [var_room for var_room in var_test_rooms
                                if var_room in preset["data"]["environment"]]
        if len(var_real_query_rooms) != len(var_test_rooms):
            var_support_rooms = var_real_query_rooms
    ## the teacher's data and the student's support/query pool must be disjoint
    if set(var_train_rooms) & set(var_test_rooms):
        raise ValueError(f"train rooms {var_train_rooms} and test rooms {var_test_rooms} overlap")
    if var_support_rooms and set(var_train_rooms) & set(var_support_rooms):
        raise ValueError(f"train rooms {var_train_rooms} and support rooms {var_support_rooms} "
                         f"overlap")
    #
    if var_args.window is not None:
        preset["hwild"]["window"] = var_args.window

    train_sets_by_room, test_sets_by_room, support_sets_by_room = {}, {}, {}
    for var_room in var_train_rooms:
        train_sets_by_room[var_room] = load_room_wimans(var_room, var_args, "train] ")
    for var_room in var_test_rooms:
        test_sets_by_room[var_room] = load_room_wimans(var_room, var_args, "test]  ")
    ## the support rooms are loaded even when they are not scored on, so the student can be given
    ## data from a room the query set never touches
    for var_room in (var_support_rooms or []):
        if var_room in test_sets_by_room:
            continue
        support_sets_by_room[var_room] = load_room_wimans(var_room, var_args, "support]")

    #
    ## -------------------------------- shared transmitter frame ---------------------------------
    ## every room's positions are already meters from the transmitter (preset["layouts"]), so one
    ## rectangle with a small margin covers them all and every model predicts in the same frame
    var_all_xy = np.concatenate([var_y for _, var_y in list(train_sets_by_room.values())
                                 + list(test_sets_by_room.values())
                                 + list(support_sets_by_room.values())])
    var_margin = 0.5
    var_rect = (float(var_all_xy[:, 0].min() - var_margin), float(var_all_xy[:, 0].max() + var_margin),
                float(var_all_xy[:, 1].min() - var_margin), float(var_all_xy[:, 1].max() + var_margin))
    print(f"\nTransmitter-anchored frame rectangle (m): x [{var_rect[0]:.2f}, {var_rect[1]:.2f}], "
          f"y [{var_rect[2]:.2f}, {var_rect[3]:.2f}]  "
          f"(span {var_rect[1] - var_rect[0]:.2f} x {var_rect[3] - var_rect[2]:.2f} m)")
    #
    if var_args.standardize == "window":
        for var_sets in (train_sets_by_room, test_sets_by_room, support_sets_by_room):
            for var_room, (var_x, var_y) in var_sets.items():
                var_sets[var_room] = (standardize_windows(var_x), var_y)
        print("CSI: per-window z-score (gain/AGC removed)")
    else:
        print("CSI: raw amplitude")
    #
    for var_sets in (train_sets_by_room, test_sets_by_room, support_sets_by_room):
        for var_room, (var_x, var_y) in var_sets.items():
            var_sets[var_room] = (var_x, normalize_xy_meters(var_y, var_rect))

    var_pool_sets_by_room = {**test_sets_by_room, **support_sets_by_room}
    var_save_path = reserve_viz_dir(path=f"few_shot_wimans_to_wimans/{'_'.join(var_train_rooms)}")
    var_results = run_few_shot_localization(
        train_sets_by_room, var_pool_sets_by_room,
        var_few_shot_ratio=var_args.few_shot_ratio, var_kd_weight=var_args.kd_weight,
        var_repeat=var_repeat, var_teacher_epochs=var_args.teacher_epochs,
        var_student_epochs=var_args.student_epochs,
        var_init_from_teacher=not var_args.no_init_from_teacher,
        var_bounds=var_rect, save_path=var_save_path,
        var_support_rooms=var_support_rooms, var_query_rooms=var_test_rooms)

    #
    ## run few-shot distillation
    var_result = {
        "protocol": ("train few-shot KD fine-tune on WiMANS (real or synthetic) -> test held-out "
                     "WiMANS"),
        "train_rooms": var_train_rooms,
        "test_rooms": var_test_rooms,
        "support_rooms": (var_support_rooms if var_support_rooms else var_test_rooms),
        "support_rooms_explicit": bool(var_support_rooms),
        "few_shot_ratio": var_args.few_shot_ratio,
        "kd_weight": var_args.kd_weight,
        "init_from_teacher": not var_args.no_init_from_teacher,
        "repeat": var_repeat,
        "teacher_epochs": (var_args.teacher_epochs if var_args.teacher_epochs is not None
                           else preset["hwild"]["epoch"]),
        "student_epochs": (var_args.student_epochs if var_args.student_epochs is not None
                           else preset["hwild"]["epoch"]),
        "window": preset["hwild"]["window"],
        "max_samples": var_args.max_samples,
        "features": var_args.features,
        "users": var_users,
        "standardize": var_args.standardize,
        "hp_overrides": var_hp_applied,
        "tx_frame_rect_m": var_rect,
        "per_env": var_results,
    }

    formatted = format_result(var_model, var_task, var_result, var_args.few_shot_ratio,
                              var_args.kd_weight, join_envs(var_train_rooms))
    ## write the JSON and the human-readable report to auto-named files, so the per-teacher-room and
    ## repeated runs never collide and never need renaming
    var_epochs_tag = (var_args.student_epochs if var_args.student_epochs is not None
                      else preset["hwild"]["epoch"])
    var_stem = build_run_stem(var_model, var_task, f"train-{join_envs(var_train_rooms)}",
                              f"test-{join_envs(var_test_rooms)}", f"k{var_args.few_shot_ratio}",
                              f"r{var_repeat}", f"e{var_epochs_tag}", run_timestamp())
    var_json_path, var_txt_path = save_run_outputs(
        preset["path"].get("save_dir", "output"), "few_shot_multi_rooms", var_stem, var_result,
        formatted)

    print(formatted)
    print(f"\nResults saved to: {var_json_path}")
    print(f"Report saved to:  {var_txt_path}")

    ## release the multi-GB per-room arrays before the process exits
    del train_sets_by_room, test_sets_by_room, support_sets_by_room
    gc.collect()


if __name__ == "__main__":
    var_start = time.time()
    run()
    print("Total time: %s seconds" % (time.time() - var_start))