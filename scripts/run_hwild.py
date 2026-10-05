"""
[file]          run_hwild.py
[description]   Leave-one-room-out (x, y) localization on the H-WILD dataset. Trains the continuous
                density head on the training rooms and evaluates the Euclidean position error in
                meters on every held-out room. See src/models/hwild_localization.py.

                Example:
                    python scripts/run_hwild.py --holdout Lounge --repeat 3 --epochs 60
                    python scripts/run_hwild.py --train_rooms Laboratory,Office,Lounge --max_files 8
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
from src.data.hwild import room_names, load_hwild_room, room_span, HWILD_ROOMS
from src.models.hwild_localization import run_hwild_cross_domain, format_hwild_result
from src.utils import save_run_outputs, build_run_stem, join_envs, run_timestamp

#
##


def parse_args():
    """
    [description]
    : parse arguments from input
    """
    #
    ##
    var_args = argparse.ArgumentParser()
    var_args.add_argument("--holdout", default=None, type=str,
                          help="held-out room to test on; the other rooms are used for training")
    var_args.add_argument("--train_rooms", default=None, type=str,
                          help="comma-separated training rooms. Overrides --holdout: every room not "
                               "listed becomes a test room.")
    var_args.add_argument("--repeat", default=3, type=int, help="number of repeated experiments")
    var_args.add_argument("--epochs", default=None, type=int,
                          help="override preset['hwild']['epoch']")
    var_args.add_argument("--window", default=None, type=int, help="window length in packets")
    var_args.add_argument("--stride", default=None, type=int, help="window hop in packets")
    var_args.add_argument("--data_root", default=None, type=str, help="override preset['hwild']['path']")
    var_args.add_argument("--max_files", default=None, type=int,
                          help="cap the captures read per room (quick runs / smoke tests)")
    var_args.add_argument("--users", default=None, type=str,
                          help="comma-separated volunteer ids, e.g. '1,2,3'")
    var_args.add_argument("--interference", default=None, type=str,
                          help="'w' (with interference), 'wo' (without), or 'w,wo'")
    #
    return var_args.parse_args()


def run():
    """
    [description]
    : run H-WILD leave-one-room-out localization
    """
    #
    ##
    SEED = 103
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    #
    var_args = parse_args()
    var_all_rooms = room_names()
    #
    if var_args.train_rooms:
        var_train_rooms = [var_room.strip() for var_room in var_args.train_rooms.split(',')]
        var_test_rooms = [var_room for var_room in var_all_rooms if var_room not in var_train_rooms]
    else:
        var_holdout = var_args.holdout if var_args.holdout else "Lounge"
        var_test_rooms = [var_holdout]
        var_train_rooms = [var_room for var_room in var_all_rooms if var_room != var_holdout]
    #
    for var_room in var_train_rooms + var_test_rooms:
        if var_room not in var_all_rooms:
            raise ValueError(f"unknown room {var_room!r}; choose from {var_all_rooms}")
    if not var_train_rooms or not var_test_rooms:
        raise ValueError("need at least one training room and one test room")
    #
    if var_args.window is not None:
        preset["hwild"]["window"] = var_args.window
    if var_args.stride is not None:
        preset["hwild"]["stride"] = var_args.stride
    if var_args.epochs is not None:
        preset["hwild"]["epoch"] = var_args.epochs
    if var_args.data_root is not None:
        preset["hwild"]["path"] = var_args.data_root
    #
    var_users = [var_user.strip() for var_user in var_args.users.split(',')] if var_args.users else None
    var_interference = ([var_i.strip() for var_i in var_args.interference.split(',')]
                        if var_args.interference else None)
    #
    ## --------------------------------------- load the rooms ----------------------------------------
    var_load_kwargs = {"var_root": var_args.data_root, "var_users": var_users,
                       "var_interference": var_interference, "var_max_files": var_args.max_files}
    train_sets_by_room, test_sets_by_room = {}, {}
    for var_room in var_train_rooms:
        var_x, var_y, var_meta = load_hwild_room(var_room, **var_load_kwargs)
        train_sets_by_room[var_room] = (var_x, var_y)
        print(f"[train] {var_room}: X {var_x.shape} XY {var_y.shape} "
              f"({var_meta['num_files']} captures, span {room_span(var_room)})")
    for var_room in var_test_rooms:
        var_x, var_y, var_meta = load_hwild_room(var_room, **var_load_kwargs)
        test_sets_by_room[var_room] = (var_x, var_y)
        print(f"[test]  {var_room}: X {var_x.shape} XY {var_y.shape} "
              f"({var_meta['num_files']} captures, span {room_span(var_room)})")
    #
    ## --------------------------------------- run the model ----------------------------------------
    var_save_path = f"./visualizations/hwild/{'_'.join(var_train_rooms)}"
    os.makedirs(var_save_path, exist_ok=True)
    var_results = run_hwild_cross_domain(train_sets_by_room, test_sets_by_room,
                                         var_repeat=var_args.repeat, save_path=var_save_path)
    #
    var_result = {
        "protocol": "leave-one-room-out",
        "train_rooms": var_train_rooms,
        "test_rooms": var_test_rooms,
        "repeat": var_args.repeat,
        "epochs": preset["hwild"]["epoch"],
        "window": preset["hwild"]["window"],
        "stride": preset["hwild"]["stride"],
        "max_files": var_args.max_files,
        "users": var_users,
        "interference": var_interference,
        "rooms_geometry": {var_room: HWILD_ROOMS[var_room] for var_room in var_all_rooms},
        "per_env": var_results,
    }
    #
    var_formatted = format_hwild_result(var_train_rooms, var_test_rooms, var_results)
    var_stem = build_run_stem("hwild", "localization",
                              f"train-{join_envs(var_train_rooms)}", f"test-{join_envs(var_test_rooms)}",
                              f"r{var_args.repeat}", f"e{preset['hwild']['epoch']}", run_timestamp())
    var_json_path, var_txt_path = save_run_outputs(
        preset["path"].get("save_dir", "output"), "hwild", var_stem, var_result, var_formatted)
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
