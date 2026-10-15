"""
[file]          run_hwild_to_wimans.py
[description]   Train the single-target (x, y) localizer (RoomLocalizer) on the H-WILD dataset and
                evaluate it on the WiMANS locations, reporting the error in meters.

                Both datasets are placed in one shared transmitter-anchored frame: origin at the
                transmitter (the H-WILD capture's AP, the WiMANS room corner), +y along the boresight
                ("away from the transmitter"), +x to the right, units of meters. WiMANS is already in
                this frame (preset["layouts"] is transmitter-anchored); H-WILD is mapped into it by
                src.data.hwild.tx_frame_xy using the AP geometry from obtain_parameters.m.

                A single shared rectangle over both datasets then normalises the positions for the
                RoomLocalizer harness, so the per-room [0, 1]^2 convention (which has no shared
                origin) is not used here; the reported error is a plain Euclidean distance in meters.

                Example:
                    python scripts/run_hwild_to_wimans.py --repeat 2 --epochs 40 --max_files 2
                    python scripts/run_hwild_to_wimans.py --standardize none
"""
#
##

import argparse
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.preset import preset
from src.data.hwild import room_names as hwild_room_names, load_hwild_room, HWILD_ROOMS
from src.data.unified_rooms import WIMANS_ROOMS, load_wimans_room_targets, normalize_xy_meters
from src.models.room_localization import run_localization_cross_domain, format_localization_result
from src.utils import save_run_outputs, build_run_stem, join_envs, run_timestamp

#
##


def parse_args():
    """
    [description]
    : command line arguments for the H-WILD -> WiMANS transfer run
    """
    #
    ##
    var_args = argparse.ArgumentParser()
    var_args.add_argument("--train_rooms", default=None, type=str,
                          help="comma-separated H-WILD training rooms (default: all four)")
    var_args.add_argument("--test_rooms", default=None, type=str,
                          help="comma-separated WiMANS test rooms (default: all three)")
    var_args.add_argument("--repeat", default=3, type=int, help="number of repeated experiments")
    var_args.add_argument("--epochs", default=None, type=int, help="override preset['hwild']['epoch']")
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
                               "scale that otherwise differs between the two datasets); 'none' feeds "
                               "raw amplitude")
    #
    return var_args.parse_args()


def standardize_windows(var_x):
    """
    [description]
    : z-score every CSI window over its time x subcarrier entries, so the amplitude level (which
      carries an arbitrary per-device gain and AGC and differs between H-WILD and WiMANS) is removed
      and only the window's frequency-selective shape is left.
    : var_x: (N, window, features) float32
    : return: same shape, float32
    """
    var_mu = var_x.mean(axis=(1, 2), keepdims=True)
    var_sd = var_x.std(axis=(1, 2), keepdims=True) + 1e-6
    return ((var_x - var_mu) / var_sd).astype(np.float32)


def run():
    """
    [description]
    : run the H-WILD -> WiMANS transfer
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
    for var_room in var_train_rooms:
        if var_room not in HWILD_ROOMS:
            raise ValueError(f"unknown H-WILD room {var_room!r}; choose from {hwild_room_names()}")
    for var_room in var_test_rooms:
        if var_room not in WIMANS_ROOMS:
            raise ValueError(f"unknown WiMANS room {var_room!r}; choose from {WIMANS_ROOMS}")
    #
    if var_args.window is not None:
        preset["hwild"]["window"] = var_args.window
    if var_args.stride is not None:
        preset["hwild"]["stride"] = var_args.stride
    if var_args.epochs is not None:
        preset["hwild"]["epoch"] = var_args.epochs
    if var_args.data_root is not None:
        preset["hwild"]["path"] = var_args.data_root
    var_users = [var_user.strip() for var_user in var_args.users.split(',')] if var_args.users else None
    var_interference = ([var_i.strip() for var_i in var_args.interference.split(',')]
                        if var_args.interference else None)
    #
    ## --------------------------- load both datasets in the TX frame -------------------------------
    ## H-WILD positions come back as meters from the capture's AP, +y along the AP boresight.
    var_load_kwargs = {"var_root": var_args.data_root, "var_users": var_users,
                       "var_interference": var_interference, "var_max_files": var_args.max_files}
    train_sets_by_room, test_sets_by_room = {}, {}
    for var_room in var_train_rooms:
        var_x, var_y, var_meta = load_hwild_room(var_room, var_frame="tx", **var_load_kwargs)
        train_sets_by_room[var_room] = (var_x, var_y)
        print(f"[train] {var_room:12s} X {var_x.shape} XY {var_y.shape} "
              f"({var_meta['num_files']} captures)")
    ## WiMANS positions are already meters from the transmitter (preset["layouts"]); only the
    ## single-occupant frames are kept, since the localizer predicts one position.
    for var_room in var_test_rooms:
        var_x, var_y = load_wimans_room_targets(var_room, var_max_samples=var_args.max_samples,
                                                var_length=preset["hwild"]["window"])
        if len(var_x) == 0:
            raise ValueError(f"no single-occupant WiMANS frames for {var_room!r}")
        test_sets_by_room[var_room] = (var_x, var_y)
        print(f"[test]  {var_room:12s} X {var_x.shape} XY {var_y.shape} "
              f"({len(np.unique(var_y.round(3), axis=0))} distinct positions)")
    #
    ## ------------------------------------ shared frame --------------------------------------------
    ## One rectangle over every loaded position defines the frame both datasets are normalised into,
    ## so a normalised offset means the same metric offset in H-WILD and WiMANS. Its origin is the
    ## transmitter, +y away, +x right (all positions are already in that frame).
    var_all_xy = np.concatenate([var_y for _, var_y in list(train_sets_by_room.values())
                                 + list(test_sets_by_room.values())])
    var_margin = 0.5
    var_rect = (float(var_all_xy[:, 0].min() - var_margin), float(var_all_xy[:, 0].max() + var_margin),
                float(var_all_xy[:, 1].min() - var_margin), float(var_all_xy[:, 1].max() + var_margin))
    var_span = (var_rect[1] - var_rect[0], var_rect[3] - var_rect[2])
    print(f"\nTransmitter-anchored frame rectangle (m): x [{var_rect[0]:.2f}, {var_rect[1]:.2f}], "
          f"y [{var_rect[2]:.2f}, {var_rect[3]:.2f}]  (span {var_span[0]:.2f} x {var_span[1]:.2f} m)")
    #
    if var_args.standardize == "window":
        for var_sets in (train_sets_by_room, test_sets_by_room):
            for var_room, (var_x, var_y) in var_sets.items():
                var_sets[var_room] = (standardize_windows(var_x), var_y)
        print("CSI: per-window z-score (gain/AGC removed)")
    else:
        print("CSI: raw amplitude")
    #
    ## normalise every room by the one shared rectangle
    for var_sets in (train_sets_by_room, test_sets_by_room):
        for var_room, (var_x, var_y) in var_sets.items():
            var_sets[var_room] = (var_x, normalize_xy_meters(var_y, var_rect))
    ## the harness converts a normalised error back to meters by multiplying by the span, so the
    ## shared rectangle's span gives the true Euclidean distance in meters for every room.
    var_spans = {var_room: var_span for var_room in list(train_sets_by_room) + list(test_sets_by_room)}
    var_bounds = {var_room: var_rect for var_room in list(train_sets_by_room) + list(test_sets_by_room)}
    #
    ## --------------------------------------- run the model ----------------------------------------
    var_save_path = f"./visualizations/hwild_to_wimans/{'_'.join(var_train_rooms)}"
    os.makedirs(var_save_path, exist_ok=True)
    var_results = run_localization_cross_domain(train_sets_by_room, test_sets_by_room,
                                                var_repeat=var_args.repeat, save_path=var_save_path,
                                                var_spans=var_spans, var_bounds=var_bounds)
    #
    var_result = {
        "protocol": "train H-WILD -> test WiMANS, shared transmitter-anchored metric frame",
        "train_rooms": var_train_rooms,
        "test_rooms": var_test_rooms,
        "repeat": var_args.repeat,
        "epochs": preset["hwild"]["epoch"],
        "window": preset["hwild"]["window"],
        "stride": preset["hwild"]["stride"],
        "max_files": var_args.max_files,
        "max_samples": var_args.max_samples,
        "users": var_users,
        "interference": var_interference,
        "standardize": var_args.standardize,
        "tx_frame_rect_m": var_rect,
        "ap_geometry": {var_room: {"ap_location": HWILD_ROOMS[var_room]["ap_location"],
                                   "ap_toward": HWILD_ROOMS[var_room]["ap_toward"]}
                        for var_room in var_train_rooms},
        "per_env": var_results,
    }
    #
    var_formatted = format_localization_result(var_train_rooms, var_test_rooms, var_results)
    var_stem = build_run_stem("hwild_to_wimans", "localization",
                              f"train-{join_envs(var_train_rooms)}", f"test-{join_envs(var_test_rooms)}",
                              f"r{var_args.repeat}", f"e{preset['hwild']['epoch']}",
                              var_args.standardize, run_timestamp())
    var_json_path, var_txt_path = save_run_outputs(
        preset["path"].get("save_dir", "output"), "hwild_to_wimans", var_stem, var_result, var_formatted)
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
