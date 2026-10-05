"""
[file]          run.py
[description]   run WiFi-based models
"""
#
##

import argparse
import random # Added
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import *
from src.data.load_data import load_data_x, load_data_y, encode_data_y, encode_occupancy_y
from src.utils import *
from configs.preset import preset

import time
import json
from datetime import datetime
from pathlib import Path

#
##
def _load_room(preset, var_task, var_model, var_users, var_env):
    """
    [description]
    : load one room's CSI amplitude and its labels for the given model.
    """
    data_pd_y = load_data_y(preset["path"]["data_y"],
                            var_environment=[var_env],
                            var_wifi_band=preset["data"]["wifi_band"],
                            var_num_users=var_users)
    var_label_list = data_pd_y["label"].to_list()
    var_x = load_data_x(preset["path"]["data_x"], var_label_list)
    if var_model in ("density_map", "density_map_dem"):
        ## Density-map group counting predicts per-location occupancy, so the label is the room's
        ## occupancy vector rather than a task encoding.
        var_y = encode_occupancy_y(data_pd_y, var_env)
    else:
        var_y = encode_data_y(data_pd_y, var_task)
        if var_model in ("AMAR_WO_RVQ", "AMAR"):
            var_y = reduce_dataset(var_y, var_task, preset["nn"]["num_obj_queries"])
    return var_x, var_y


def master_splitter(preset, var_task, var_model, var_users, var_train_envs):
    """
    [description]
    : build the per-room training sets and the per-room test sets. Pass one training room for the
      one-room protocol, or two for leave-one-room-out (train on two, test on the third). Every room
      not in var_train_envs becomes a test room.
    : return: train_sets_by_env, test_sets_by_env (dict {env_name: (X, y)})
    """
    var_all_envs = list(preset["data"]["environment"])
    for var_env in var_train_envs:
        if var_env not in var_all_envs:
            raise ValueError(f"training room {var_env!r} is not in preset['data']['environment'] "
                             f"{var_all_envs}")
    train_sets_by_env = {var_env: _load_room(preset, var_task, var_model, var_users, var_env)
                         for var_env in var_train_envs}
    test_sets_by_env = {var_env: _load_room(preset, var_task, var_model, var_users, var_env)
                        for var_env in var_all_envs if var_env not in var_train_envs}
    return train_sets_by_env, test_sets_by_env

def parse_args():
    """
    [description]
    : parse arguments from input
    """
    #
    ##
    var_args = argparse.ArgumentParser()
    #
    var_args.add_argument("--model", default = preset["model"], type = str)
    var_args.add_argument("--task", default = preset["task"], type = str)
    var_args.add_argument("--repeat", default = preset["repeat"], type = int)
    var_args.add_argument("--users", default="0, 1,2,3,4,5", type=str, help="Comma-separated list of user IDs")
    var_args.add_argument("--env", default="empty_room", type=str, help="training room name")
    var_args.add_argument("--train_envs", default=None, type=str,
                          help="Comma-separated training rooms. Overrides --env. Pass two rooms for "
                               "leave-one-room-out: train on those two and test on the remaining one. "
                               "Every room not listed becomes a test room.")
    var_args.add_argument("--epochs", default=None, type=int,
                          help="Override preset['nn']['epoch']. The cosine LR schedule is tied to "
                               "this value, so it must match the actual training length.")
    #
    return var_args.parse_args()


def format_result(var_model, var_task, result):
    """
    Build a formatted string of the results.
    """
    # (avg_key, se_key, label) in display order
    metric_specs = [
        ("avg_accuracy",    "se_accuracy",    "Avg Accuracy"),
        ("avg_precision",   "se_precision",   "Avg Precision"),
        ("avg_recall",      "se_recall",      "Avg Recall"),
        ("avg_f1_score",    "se_f1_score",    "Avg F1 Score"),
        ("avg_PPP",         "se_PPP",         "Avg Perfect Prediction %"),
        ("avg_total_error", "se_total_error", "Avg Total Error"),
    ]

    lines = ["=" * 80]
    lines.append(f"EXPERIMENT RESULTS - Model: {var_model}, Task: {var_task}")
    lines.append("=" * 80)

    per_env = result.get("per_env") if isinstance(result, dict) else None

    if isinstance(per_env, dict) and per_env:
        lines.append("PER_ENV_RESULTS:")
        for env_key, env_results in per_env.items():
            lines.append(f"\n{env_key}:")
            if 'avg_mae' in env_results:
                ## density-map group-count metrics
                lines.append("  GROUP-COUNT METRICS:")
                lines.append(f"  Exact-count Accuracy: {env_results['avg_accuracy']:.4f} ± {env_results['se_accuracy']:.4f} (SE)")
                if 'avg_balanced_accuracy' in env_results:
                    lines.append(f"  Balanced-count Accuracy: {env_results['avg_balanced_accuracy']:.4f} "
                                 f"± {env_results['se_balanced_accuracy']:.4f} (SE)")
                lines.append(f"  Count MAE: {env_results['avg_mae']:.4f} ± {env_results['se_mae']:.4f} (SE)")
                lines.append(f"  Occupancy Accuracy: {env_results['avg_occupancy_accuracy']:.4f} ± {env_results['se_occupancy_accuracy']:.4f} (SE)")
                lines.append(f"  Occupancy F1: {env_results['avg_occupancy_f1']:.4f} ± {env_results['se_occupancy_f1']:.4f} (SE)")
                per_class = env_results.get('per_class_accuracy', {})
                if per_class:
                    lines.append("  Per-count Accuracy: "
                                 + ", ".join(f"{k}:{v:.3f}" for k, v in per_class.items()))
            else:
                for avg_key, se_key, label in metric_specs:
                    if avg_key in env_results:
                        se = env_results.get(se_key, float("nan"))
                        lines.append(f"  {label}: {env_results[avg_key]:.4f} ± {se:.4f} (SE)")
    elif isinstance(result, dict):
        lines.append("SINGLE MODEL RESULTS:")
        has_aggregated = any(avg_key in result for avg_key, _, _ in metric_specs)
        if has_aggregated:
            for avg_key, se_key, label in metric_specs:
                if avg_key in result:
                    se = result.get(se_key, float("nan"))
                    lines.append(f"  {label}: {result[avg_key]:.4f} ± {se:.4f} (SE)")
        elif "precision" in result:
            # Raw single-run results (no averaging across repeats)
            single_specs = [
                ("precision", "Precision"),
                ("recall", "Recall"),
                ("perfect_prediction_percentage", "Perfect Prediction %"),
                ("f1_score", "F1 Score"),
                ("accuracy", "Accuracy"),
                ("total_error", "Total Error"),
            ]
            for key, label in single_specs:
                if key in result:
                    lines.append(f"  {label}: {result[key]:.4f}")

    return "\n".join(lines)

#
##
def run():
    """
    [description]
    : run WiFi-based models
    """
    SEED = 103 # Ensuring the results are reproducible
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    #
    ## parse arguments from input
    var_args = parse_args()
    #
    var_task = var_args.task
    var_model = var_args.model
    var_repeat = var_args.repeat
    var_users = [u.strip() for u in var_args.users.split(',')]
    var_env = var_args.env
    var_train_envs = ([e.strip() for e in var_args.train_envs.split(',')]
                      if var_args.train_envs else [var_env])
    if var_args.epochs is not None:
        preset["nn"]["epoch"] = var_args.epochs

    # Ensuring there is no data leakage while doing splits.
    train_sets_by_env, test_sets_by_env = master_splitter(
        preset, var_task, var_model, var_users, var_train_envs)

    ## the run directory is named after the training rooms so one-room and leave-one-room-out runs
    ## do not collide
    save_path=Path(f'./visualizations/cross_domain/{"_".join(var_train_envs)}/1')
    while save_path.is_dir():
        new_name = str((int(save_path.name)+1))
        save_path = save_path.parent / new_name
    #
    ## run WiFi-based model
    if var_model == "density_map":
        all_envs_results = run_density_map_cross_domain(train_sets_by_env, test_sets_by_env,
                                                        var_repeat, var_task, save_path)
    elif var_model == "density_map_dem":
        all_envs_results = run_density_map_dem_cross_domain(train_sets_by_env, test_sets_by_env,
                                                            var_repeat, var_task, save_path)
    else:
        data_x_train = np.concatenate([var_x for var_x, _ in train_sets_by_env.values()])
        data_y_train = np.concatenate([var_y for _, var_y in train_sets_by_env.values()])
        all_envs_results = run_cross_domain(data_x_train, data_y_train, test_sets_by_env,
                                            var_repeat, var_task, "_".join(var_train_envs), save_path)
    #
    ##
    result = {
    "per_env": all_envs_results,
    "model": var_model,
    "task": var_task,
    "repeat": var_repeat,
    "train_envs": var_train_envs,
    "test_envs": [e for e in preset["data"]["environment"] if e not in var_train_envs],
    "epochs": preset["nn"]["epoch"],
    "data": preset["data"],
    "nn": preset["nn"],
    }

    #
    ## write the JSON and the human-readable report to auto-named files, so the one-room and
    ## leave-one-room-out folds (and repeated runs) never collide and never need renaming
    formatted = format_result(var_model, var_task, result)
    var_test_envs = [e for e in preset["data"]["environment"] if e not in var_train_envs]
    var_stem = build_run_stem(var_model, var_task,
                              f"train-{join_envs(var_train_envs)}", f"test-{join_envs(var_test_envs)}",
                              f"r{var_repeat}", f"e{preset['nn']['epoch']}", run_timestamp())
    var_json_path, var_txt_path = save_run_outputs(
        preset["path"].get("save_dir", "output"), "cross_domain", var_stem, result, formatted)

    print(formatted)
    print(f"\nResults saved to: {var_json_path}")
    print(f"Report saved to:  {var_txt_path}")

if __name__ == "__main__":
    #
    ##
    start_time = time.time()
    run()
    print("Total time: %s seconds" % (time.time() - start_time))