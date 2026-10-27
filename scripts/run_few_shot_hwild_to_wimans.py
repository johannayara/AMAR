"""
[file]          run_few_shot_hwild_to_wimans.py
[description]   Few-shot knowledge distillation for train -> test localization.

                Teacher : RoomLocalizer trained on every train rooms, then frozen.
                Student : RoomLocalizer warm-started from the teacher and fine-tuned on a small
                          fraction of the test frames (--few_shot_ratio) with the supervised Huber
                          position loss plus --kd_weight * a smooth-L1 match to the frozen teacher.
                Test    : the test frames held out of both the support set and the student's
                          validation slice, so the reported error is on frames neither model was
                          tuned on. Teacher and constant-position baseline are reported alongside.

                Both datasets sit in the shared transmitter-anchored frame (see
                src/data/hwild.tx_frame_xy and src/data/unified_rooms.load_wimans_room_targets), so
                the positions are directly comparable in meters.


                Example:
                    python scripts/run_few_shot_hwild_to_wimans.py --few_shot_ratio 0.05 --kd_weight 1.0
                    python scripts/run_few_shot_hwild_to_wimans.py --few_shot_ratio 0.05 --kd_weight 0
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
from src.data.hwild import room_names as hwild_room_names, load_hwild_room, HWILD_ROOMS
from src.data.unified_rooms import WIMANS_ROOMS, load_wimans_room_targets, normalize_xy_meters
from src.models.room_localization import run_few_shot_localization, format_few_shot_result
from src.utils import save_run_outputs, build_run_stem, join_envs, run_timestamp, reserve_viz_dir

#
##


def parse_args():
    """
    [description]
    : command line arguments for the H-WILD -> WiMANS few-shot run
    """
    #
    ##
    var_args = argparse.ArgumentParser()
    var_args.add_argument("--train_rooms", default=None, type=str,
                          help="comma-separated teacher rooms (default: H-WILD)")
    var_args.add_argument("--test_rooms", default=None, type=str,
                          help="comma-separated rooms forming the support/query pool "
                               "(default: WiMANS)")
    var_args.add_argument("--few_shot_ratio", default=0.05, type=float,
                          help="fraction of the WiMANS pool used to fine-tune the student")
    var_args.add_argument("--kd_weight", default=1.0, type=float,
                          help="weight of the teacher distillation term (0 = supervised few-shot only)")
    var_args.add_argument("--repeat", default=3, type=int, help="number of repeated experiments")
    var_args.add_argument("--teacher_epochs", default=None, type=int,
                          help="override the teacher's epochs (default preset['hwild']['epoch'])")
    var_args.add_argument("--student_epochs", default=None, type=int,
                          help="override the student's epochs (default preset['hwild']['epoch'])")
    var_args.add_argument("--no_init_from_teacher", action="store_true",
                          help="start the student from random weights instead of the teacher's")
    var_args.add_argument("--window", default=None, type=int, help="window length in packets")
    var_args.add_argument("--stride", default=None, type=int, help="window hop in packets")
    var_args.add_argument("--data_root", default=None, type=str, help="override preset['hwild']['path']")
    var_args.add_argument("--max_files", default=None, type=int,
                          help="cap the H-WILD captures read per room (quick runs / smoke tests)")
    var_args.add_argument("--max_samples", default=None, type=int,
                          help="cap the WiMANS samples drawn per room")
    var_args.add_argument("--users", default=None, type=str,
                          help="comma-separated H-WILD volunteer ids, e.g. '1,2,3'")
    var_args.add_argument("--interference", default=None, type=str,
                          help="H-WILD 'w' (with interference), 'wo' (without), or 'w,wo'")
    var_args.add_argument("--standardize", default="window", choices=("none", "window"),
                          help="'window' z-scores each CSI window (removes the per-capture gain/AGC "
                               "scale that differs between the two datasets); 'none' feeds raw amplitude")
    #
    return var_args.parse_args()


def standardize_windows(var_x):
    """
    [description]
    : z-score every CSI window over its time x subcarrier entries, removing the per-device gain/AGC
      level so only the window's frequency-selective shape is left.
    : var_x: (N, window, features) float32
    : return: same shape, float32
    """
    var_mu = var_x.mean(axis=(1, 2), keepdims=True)
    var_sd = var_x.std(axis=(1, 2), keepdims=True) + 1e-6
    return ((var_x - var_mu) / var_sd).astype(np.float32)





def run():
    """
    [description]
    : run the H-WILD -> WiMANS few-shot distillation
    """
    #
    ##
    SEED = 103
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    #
    var_args = parse_args()
    var_train_rooms = ([var_room.strip() for var_room in var_args.train_rooms.split(',')]
                       if var_args.train_rooms else hwild_room_names())
    var_test_rooms = ([var_room.strip() for var_room in var_args.test_rooms.split(',')]
                      if var_args.test_rooms else list(WIMANS_ROOMS))
    ## the teacher's data and the student's support/query pool must be disjoint, or the "held-out"
    ## query frames would be frames the teacher was trained on
    if set(var_train_rooms) & set(var_test_rooms):
        raise ValueError(f"train rooms {var_train_rooms} and test rooms {var_test_rooms} overlap")
    #
    if var_args.window is not None:
        preset["hwild"]["window"] = var_args.window
    if var_args.stride is not None:
        preset["hwild"]["stride"] = var_args.stride
    if var_args.data_root is not None:
        preset["hwild"]["path"] = var_args.data_root
    var_users = [var_user.strip() for var_user in var_args.users.split(',')] if var_args.users else None
    var_interference = ([var_i.strip() for var_i in var_args.interference.split(',')]
                        if var_args.interference else None)
    #
    ## --------------------------- load both datasets in the TX frame -------------------------------
    var_load_kwargs = {"var_root": var_args.data_root, "var_users": var_users,
                       "var_interference": var_interference, "var_max_files": var_args.max_files}
    train_sets_by_room, test_sets_by_room = {}, {}
    for var_room in var_train_rooms:
        if var_room in hwild_room_names():
            var_x, var_y, var_meta = load_hwild_room(var_room, var_frame="tx", **var_load_kwargs)
            train_sets_by_room[var_room] = (var_x, var_y)
            print(f"[train] {var_room:12s} X {var_x.shape} XY {var_y.shape} "
                f"({var_meta['num_files']} captures)")
        elif var_room in WIMANS_ROOMS:
            var_x, var_y = load_wimans_room_targets(var_room, var_max_samples=var_args.max_samples,
                                                    var_length=preset["hwild"]["window"])
            if len(var_x) == 0:
                raise ValueError(f"no single-occupant WiMANS frames for {var_room!r}")
            train_sets_by_room[var_room] = (var_x, var_y)
            print(f"[train] {var_room:12s} X {var_x.shape} XY {var_y.shape} "
                  f"({len(np.unique(var_y.round(3), axis=0))} distinct positions)")
    ## WiMANS positions are already meters from the transmitter (preset["layouts"]); only the
    ## single-occupant frames are kept, since the localizer predicts one position.
    for var_room in var_test_rooms:
        if var_room in hwild_room_names():
            var_x, var_y, var_meta = load_hwild_room(var_room, var_frame="tx", **var_load_kwargs)
            test_sets_by_room[var_room] = (var_x, var_y)
            print(f"[test]  {var_room:12s} X {var_x.shape} XY {var_y.shape} "
                f"({var_meta['num_files']} captures)")
        elif var_room in WIMANS_ROOMS:
            var_x, var_y = load_wimans_room_targets(var_room, var_max_samples=var_args.max_samples,
                                                    var_length=preset["hwild"]["window"])
            if len(var_x) == 0:
                raise ValueError(f"no single-occupant WiMANS frames for {var_room!r}")
            test_sets_by_room[var_room] = (var_x, var_y)
            print(f"[test]  {var_room:12s} X {var_x.shape} XY {var_y.shape} "
                f"({len(np.unique(var_y.round(3), axis=0))} distinct positions)")
    #
    ## ------------------------------------ shared frame --------------------------------------------
    var_all_xy = np.concatenate([var_y for _, var_y in list(train_sets_by_room.values())
                                 + list(test_sets_by_room.values())])
    var_margin = 0.5
    var_rect = (float(var_all_xy[:, 0].min() - var_margin), float(var_all_xy[:, 0].max() + var_margin),
                float(var_all_xy[:, 1].min() - var_margin), float(var_all_xy[:, 1].max() + var_margin))
    print(f"\nTransmitter-anchored frame rectangle (m): x [{var_rect[0]:.2f}, {var_rect[1]:.2f}], "
          f"y [{var_rect[2]:.2f}, {var_rect[3]:.2f}]  "
          f"(span {var_rect[1] - var_rect[0]:.2f} x {var_rect[3] - var_rect[2]:.2f} m)")
    #
    if var_args.standardize == "window":
        for var_sets in (train_sets_by_room, test_sets_by_room):
            for var_room, (var_x, var_y) in var_sets.items():
                var_sets[var_room] = (standardize_windows(var_x), var_y)
        print("CSI: per-window z-score (gain/AGC removed)")
    else:
        print("CSI: raw amplitude")
    #
    for var_sets in (train_sets_by_room, test_sets_by_room):
        for var_room, (var_x, var_y) in var_sets.items():
            var_sets[var_room] = (var_x, normalize_xy_meters(var_y, var_rect))
    #
    ## --------------------------------------- run the model ----------------------------------------
    var_save_path = reserve_viz_dir(path=f"few_shot_hwild_to_wimans/{'_'.join(var_train_rooms)}")
    var_results = run_few_shot_localization(
        train_sets_by_room, test_sets_by_room,
        var_few_shot_ratio=var_args.few_shot_ratio, var_kd_weight=var_args.kd_weight,
        var_repeat=var_args.repeat, var_teacher_epochs=var_args.teacher_epochs,
        var_student_epochs=var_args.student_epochs,
        var_init_from_teacher=not var_args.no_init_from_teacher,
        var_bounds=var_rect, save_path=var_save_path)
    #
    var_result = {
        "protocol": "train H-WILD (teacher) -> few-shot KD fine-tune on WiMANS -> test held-out WiMANS",
        "train_rooms": var_train_rooms,
        "test_rooms": var_test_rooms,
        "few_shot_ratio": var_args.few_shot_ratio,
        "kd_weight": var_args.kd_weight,
        "init_from_teacher": not var_args.no_init_from_teacher,
        "repeat": var_args.repeat,
        "teacher_epochs": (var_args.teacher_epochs if var_args.teacher_epochs is not None
                           else preset["hwild"]["epoch"]),
        "student_epochs": (var_args.student_epochs if var_args.student_epochs is not None
                           else preset["hwild"]["epoch"]),
        "window": preset["hwild"]["window"],
        "stride": preset["hwild"]["stride"],
        "max_files": var_args.max_files,
        "max_samples": var_args.max_samples,
        "users": var_users,
        "interference": var_interference,
        "standardize": var_args.standardize,
        "tx_frame_rect_m": var_rect,
        "ap_geometry": {
            var_room: ({"ap_location": HWILD_ROOMS[var_room]["ap_location"],
                        "ap_toward": HWILD_ROOMS[var_room]["ap_toward"]}
                    if var_room in hwild_room_names()
                    else {"ap_location": (0.0, 0.0), "ap_toward": (0.0, 0.0)})
            for var_room in var_train_rooms
        },
        "per_env": var_results,
    }
    #
    var_formatted = format_few_shot_result(var_args.few_shot_ratio, var_args.kd_weight, var_results)
    var_stem = build_run_stem("few_shot_hwild_to_wimans", "localization",
                              f"train-{join_envs(var_train_rooms)}", f"test-{join_envs(var_test_rooms)}",
                              f"fs{var_args.few_shot_ratio}", f"kd{var_args.kd_weight}",
                              f"r{var_args.repeat}", var_args.standardize, run_timestamp())
    var_json_path, var_txt_path = save_run_outputs(
        preset["path"].get("save_dir", "output"), "few_shot_hwild_to_wimans", var_stem,
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
