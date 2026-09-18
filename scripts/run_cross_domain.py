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
from src.data.load_data import load_data_x, load_data_y, encode_data_y
from src.utils import *
from configs.preset import preset

import time
import json
from datetime import datetime
from pathlib import Path

#
##
def master_splitter(preset, var_task, var_model, var_users, var_env="empty_room"):
    data_pd_y = load_data_y(preset["path"]["data_y"],
                             var_environment=[var_env],
                             var_wifi_band=preset["data"]["wifi_band"],
                             var_num_users=var_users)
    var_label_list = data_pd_y["label"].to_list()
    data_x_train = load_data_x(preset["path"]["data_x"], var_label_list)
    data_y_train = encode_data_y(data_pd_y, var_task)

    if var_model in ("AMAR_WO_RVQ", "AMAR"):
        data_y_train = reduce_dataset(data_y_train, var_task, preset["nn"]["num_obj_queries"])

    test_sets_by_env = {}
    other_envs = [e for e in preset["data"]["environment"] if e != var_env]
    for e in other_envs:
        data_pd_y_test = load_data_y(preset["path"]["data_y"],
                                      var_environment=[e],
                                      var_wifi_band=preset["data"]["wifi_band"],
                                      var_num_users=var_users)
        var_label_list_test = data_pd_y_test["label"].to_list()
        X_test_e = load_data_x(preset["path"]["data_x"], var_label_list_test)
        y_test_e = encode_data_y(data_pd_y_test, var_task)

        if var_model in ("AMAR_WO_RVQ", "AMAR"):
            y_test_e = reduce_dataset(y_test_e, var_task, preset["nn"]["num_obj_queries"])

        test_sets_by_env[e] = (X_test_e, y_test_e)

    return data_x_train, data_y_train, test_sets_by_env

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
    var_args.add_argument("--env", default="empty_room", type=str, help="room name")
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

def save_result(var_model, var_task, var_repeat, result):
    """
    [description]
    : save the full result dict to one JSON file per run
    """
    result["model"] = var_model
    result["task"] = var_task
    result["repeat"] = var_repeat
    result["data"] = preset["data"]
    result["nn"] = preset["nn"]
    result["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    save_dir = preset["path"].get("save_dir", "output")
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        save_dir,
        f"result_{var_model}_{var_task}_r{var_repeat}_{timestamp}.json",
    )

    with open(out_path, "w") as f:
        json.dump(result, f, indent=4, cls=NumpyEncoder)

    return out_path


def write_result(out_path, formatted):
    """
    [description]
    : append the formatted results to the run's output file
    """
    with open(out_path, "a") as f:
        f.write(formatted + "\n")
        f.write("\nFull Result Details:\n")
        f.write(json.dumps(json.loads(open(out_path).read()) if False else "", default=str))
    return out_path

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

    # Ensuring there is no data leakage while doing splits.
    data_x_train, data_y_train, test_sets_by_env = master_splitter(preset, var_task, var_model, var_users, var_env)

    save_path=Path(f'./visualizations/cross_domain/{var_env}/1')
    while save_path.is_dir():
        new_name = str((int(save_path.name)+1))
        save_path = save_path.parent / new_name
    #
    ## run WiFi-based model
    all_envs_results = run_cross_domain(data_x_train, data_y_train, test_sets_by_env, var_repeat, var_task, var_env, save_path)
    #
    ##
    result = {
    "per_env": all_envs_results,
    "model": var_model,
    "task": var_task,
    "repeat": var_repeat,
    "data": preset["data"],
    "nn": preset["nn"],
    }

    
    # Also write a human-readable summary alongside the JSON
    formatted = format_result(var_model, var_task, result)
    print(formatted)

if __name__ == "__main__":
    #
    ##
    start_time = time.time()
    run()
    print("Total time: %s seconds" % (time.time() - start_time))