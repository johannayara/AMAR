"""
[file]          run_lor_all.py
[description]   Multi-user room-agnostic leave-one-room-out localization over the pooled WiMANS +
                H-WILD room set. Loads all rooms of both datasets into one shared [0, 1]^2 frame with
                each sample's label kept as a SET of positions (WiMANS: one per occupied user, H-WILD:
                the single person), holds out one room (random by default, or every room with
                --folds all), trains the density head on the rest and reports the multi-user position
                error in meters on the held-out room.

                Examples:
                    python scripts/run_lor_all.py --max_per_room 300 --epochs 40 --repeat 2
                    python scripts/run_lor_all.py --folds all --max_per_room 200 --epochs 30
                    python scripts/run_lor_all.py --holdout Lounge --max_per_room 100 --epochs 5
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
from src.data.unified_rooms import (unified_room_names, dataset_of, load_unified_room,
                                    room_span, room_bounds)
from src.models.room_density import run_room_density_cross_domain
from src.models.set_localization import run_set_localization_cross_domain
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
    var_args.add_argument("--model", default="set", type=str, choices=("set", "density"),
                          help="localization head: 'set' (general set prediction, default) or "
                               "'density' (continuous density map)")
    var_args.add_argument("--holdout", default="random", type=str,
                          help="held-out room name, or 'random' to draw one (seeded by --seed)")
    var_args.add_argument("--folds", default=None, type=str,
                          help="'all' to hold out every room in turn; otherwise one fold")
    var_args.add_argument("--repeat", default=2, type=int, help="repeated experiments per fold")
    var_args.add_argument("--epochs", default=None, type=int, help="override preset['nn']['epoch']")
    var_args.add_argument("--max_per_room", default=300, type=int,
                          help="cap on samples/windows read per room (memory + runtime)")
    var_args.add_argument("--length", default=None, type=int, help="window length in timesteps")
    var_args.add_argument("--seed", default=103, type=int, help="seed for the random holdout and training")
    var_args.add_argument("--data_root", default=None, type=str, help="override preset['hwild']['path']")
    return var_args.parse_args()


def load_all_rooms(var_args):
    """
    [description]
    : load every room of both datasets into the shared frame.
    : return: dict room -> (X, POS, MASK)
    """
    #
    var_kwargs = {"var_root": var_args.data_root} if var_args.data_root else {}
    ## 0 (or negative) means no cap: use every sample/window of the room.
    var_max_samples = var_args.max_per_room if var_args.max_per_room and var_args.max_per_room > 0 else None
    var_sets = {}
    for var_room in unified_room_names():
        var_x, var_pos, var_mask = load_unified_room(var_room, var_max_samples=var_max_samples,
                                                     var_length=var_args.length, **var_kwargs)
        var_sets[var_room] = (var_x, var_pos, var_mask)
        var_counts = var_mask.sum(axis=1)
        print(f"  loaded {dataset_of(var_room):6s} {var_room:12s} X {var_x.shape} "
              f"users/sample {var_counts.mean():.2f} span {tuple(round(v, 2) for v in room_span(var_room))}")
    return var_sets


def resolve_folds(var_args):
    """
    [description]
    : the list of held-out rooms. "all" holds out every room in turn; otherwise a single fold, drawn
      at random (seeded) when --holdout is "random".
    """
    #
    var_rooms = unified_room_names()
    if var_args.folds == "all":
        return list(var_rooms)
    if var_args.holdout == "random":
        return [random.Random(var_args.seed).choice(var_rooms)]
    if var_args.holdout not in var_rooms:
        raise ValueError(f"unknown room {var_args.holdout!r}; choose from {var_rooms}")
    return [var_args.holdout]


def run():
    """
    [description]
    : run the pooled multi-user leave-one-room-out localization
    """
    #
    ##
    var_args = parse_args()
    random.seed(var_args.seed)
    np.random.seed(var_args.seed)
    torch.manual_seed(var_args.seed)
    if var_args.epochs is not None:
        preset["nn"]["epoch"] = var_args.epochs
    if var_args.length is not None:
        preset["hwild"]["window"] = var_args.length
    #
    var_rooms = unified_room_names()
    var_spans = {var_room: room_span(var_room) for var_room in var_rooms}
    var_bounds = {var_room: room_bounds(var_room) for var_room in var_rooms}
    var_folds = resolve_folds(var_args)
    #
    print("Loading rooms:")
    var_sets = load_all_rooms(var_args)
    print(f"Folds (held-out rooms): {var_folds}")
    #
    ## ------------------------------------------ folds -----------------------------------------------
    var_fold_results = {}
    for var_holdout in var_folds:
        var_train_rooms = [var_room for var_room in var_rooms if var_room != var_holdout]
        print(f"\n{'=' * 88}\nFold: held-out {var_holdout} ({dataset_of(var_holdout)}) | "
              f"train {var_train_rooms}\n{'=' * 88}")
        var_train = {var_room: var_sets[var_room] for var_room in var_train_rooms}
        var_test = {var_holdout: var_sets[var_holdout]}
        var_save_path = f"./visualizations/lor_all/holdout_{var_holdout}"
        os.makedirs(var_save_path, exist_ok=True)
        var_runner = run_set_localization_cross_domain if var_args.model == "set" \
            else run_room_density_cross_domain
        var_results = var_runner(
            var_train, var_test, var_repeat=var_args.repeat, save_path=var_save_path,
            var_spans=var_spans, var_bounds=var_bounds)
        var_fold_results[var_holdout] = var_results[var_holdout]
    #
    ## ---------------------------------------- summary -----------------------------------------------
    var_lines = ["=" * 96,
                 f"POOLED MULTI-USER ROOM-AGNOSTIC LOR ({var_args.model}) - WiMANS + H-WILD | "
                 f"folds {var_folds} | repeat {var_args.repeat} | epochs {preset['nn']['epoch']} | "
                 f"max_per_room {var_args.max_per_room}",
                 "=" * 96,
                 f"{'room':12s} {'dataset':7s} {'setErr':>8s} {'baseline':>9s} {'matchedMDE':>11s} "
                 f"{'cntMAE':>7s} {'exactCnt':>9s} {'detF1':>7s}"]
    var_model_all, var_base_all = [], []
    for var_holdout, var_res in var_fold_results.items():
        var_lines.append(
            f"{var_holdout:12s} {dataset_of(var_holdout):7s} "
            f"{var_res['avg_mean_error_m']:8.3f} {var_res['avg_const_mean_error_m']:9.3f} "
            f"{var_res['avg_mde_matched_m']:11.3f} {var_res['avg_count_mae']:7.3f} "
            f"{var_res['avg_exact_count_acc']:9.3f} {var_res['avg_detection_f1']:7.3f}")
        var_model_all.append(var_res["avg_mean_error_m"])
        var_base_all.append(var_res["avg_const_mean_error_m"])
    var_lines.append("-" * 96)
    var_lines.append(f"{'AVERAGE':12s} {'':7s} {np.mean(var_model_all):8.3f} "
                     f"{np.mean(var_base_all):9.3f}")
    var_lines.append("\nsetErr = mean matched distance + per-miss penalty (multi-user). Beating the "
                     "baseline column means the model learned a transferable mapping.")
    var_formatted = "\n".join(var_lines)
    #
    var_result = {
        "protocol": f"pooled multi-user leave-one-room-out (WiMANS + H-WILD, {var_args.model} head)",
        "model": var_args.model,
        "folds": var_folds,
        "rooms": var_rooms,
        "datasets": {var_room: dataset_of(var_room) for var_room in var_rooms},
        "repeat": var_args.repeat,
        "epochs": preset["nn"]["epoch"],
        "window": preset["hwild"]["window"],
        "max_per_room": var_args.max_per_room,
        "seed": var_args.seed,
        "spans": {var_room: list(var_spans[var_room]) for var_room in var_rooms},
        "per_fold": var_fold_results,
        "avg_model_mean_error_m": float(np.mean(var_model_all)),
        "avg_const_mean_error_m": float(np.mean(var_base_all)),
    }
    var_stem = build_run_stem("lor_all", f"multi_user_{var_args.model}",
                              f"folds-{join_envs(var_folds)}",
                              f"r{var_args.repeat}", f"e{preset['nn']['epoch']}", run_timestamp())
    var_json_path, var_txt_path = save_run_outputs(
        preset["path"].get("save_dir", "output"), "lor_all", var_stem, var_result, var_formatted)
    #
    print("\n" + var_formatted)
    print(f"\nResults saved to: {var_json_path}")
    print(f"Report saved to:  {var_txt_path}")


if __name__ == "__main__":
    #
    ##
    var_start = time.time()
    run()
    print("Total time: %s seconds" % (time.time() - var_start))
