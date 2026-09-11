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
    [description]
    : build a formatted string of the results
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"EXPERIMENT RESULTS - Model: {var_model}, Task: {var_task}")
    lines.append("=" * 80)

    if isinstance(result, dict) and any(
        isinstance(key, str) and key.startswith("layer_") for key in result
    ):
        lines.append("LAYERED MODEL RESULTS:")
        for layer_key in sorted(
            [k for k in result.keys() if isinstance(k, str) and k.startswith("layer_")]
        ):
            layer_results = result[layer_key]
            lines.append(f"\n{layer_key.upper()}:")
            if 'avg_precision' in layer_results:
                lines.append(f"  Avg Precision: {layer_results['avg_precision']:.4f} ± {layer_results['se_precision']:.4f} (SE)")
            if 'avg_recall' in layer_results:
                lines.append(f"  Avg Recall: {layer_results['avg_recall']:.4f} ± {layer_results['se_recall']:.4f} (SE)")
            if 'avg_PPP' in layer_results:
                lines.append(f"  Avg Perfect Prediction %: {layer_results['avg_PPP']:.4f} ± {layer_results['se_PPP']:.4f} (SE)")
            if 'avg_f1_score' in layer_results:
                lines.append(f"  Avg F1 Score: {layer_results['avg_f1_score']:.4f} ± {layer_results['se_f1_score']:.4f} (SE)")
            if 'avg_accuracy' in layer_results:
                lines.append(f"  Avg Accuracy: {layer_results['avg_accuracy']:.4f} ± {layer_results['se_accuracy']:.4f} (SE)")
            if 'avg_total_error' in layer_results:
                lines.append(f"  Avg Total Error: {layer_results['avg_total_error']:.4f} ± {layer_results['se_total_error']:.4f} (SE)")
    else:
        lines.append("SINGLE MODEL RESULTS:")
        if isinstance(result, dict):
            if 'avg_precision' in result:
                lines.append(f"  Avg Precision: {result['avg_precision']:.4f} ± {result['se_precision']:.4f} (SE)")
            if 'avg_recall' in result:
                lines.append(f"  Avg Recall: {result['avg_recall']:.4f} ± {result['se_recall']:.4f} (SE)")
            if 'avg_PPP' in result:
                lines.append(f"  Avg Perfect Prediction %: {result['avg_PPP']:.4f} ± {result['se_PPP']:.4f} (SE)")
            if 'avg_f1_score' in result:
                lines.append(f"  Avg F1 Score: {result['avg_f1_score']:.4f} ± {result['se_f1_score']:.4f} (SE)")
            if 'avg_accuracy' in result:
                lines.append(f"  Avg Accuracy: {result['avg_accuracy']:.4f} ± {result['se_accuracy']:.4f} (SE)")
            if 'avg_total_error' in result:
                lines.append(f"  Avg Total Error: {result['avg_total_error']:.4f} ± {result['se_total_error']:.4f} (SE)")
            elif 'precision' in result:
                lines.append(f"  Precision: {result['precision']:.4f}")
                if 'recall' in result:
                    lines.append(f"  Recall: {result['recall']:.4f}")
                if 'perfect_prediction_percentage' in result:
                    lines.append(f"  Perfect Prediction %: {result['perfect_prediction_percentage']:.4f}")
                if 'f1_score' in result:
                    lines.append(f"  F1 Score: {result['f1_score']:.4f}")
                if 'accuracy' in result:
                    lines.append(f"  Accuracy: {result['accuracy']:.4f}")
                if 'total_error' in result:
                    lines.append(f"  Total Error: {result['total_error']:.4f}")

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
    #

    #
    if var_model == "BCE_ABLSTM": run_model = run_bce_ablstm
    #
    elif var_model == "DEM_ABLSTM": run_model = run_dem_ablstm
    #
    elif var_model == "BCE_THAT": run_model = run_bce_that
    #    #
    elif var_model == "DEM_THAT": run_model = run_DEM_THAT

    elif var_model == "AMAR_WO_RVQ": run_model = run_cross_domain

    elif var_model == "AMAR": run_model = run_AMAR
    
    elif var_model == "multi_senseX": run_model = run_multi_senseX

    else:
        raise Exception("Not valid name for model")

    save_path=Path(f'./visualizations/{var_env}/cross_domain/1')
    if save_path.is_dir():
        new_name = str((int(save_path.name)+1))
        save_path = save_path.parent / new_name
    #
    ## run WiFi-based model
    all_envs_results = run_model(data_x_train, data_y_train, test_sets_by_env, var_repeat, var_task, var_env, save_path)
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

    # Save result dict to a per-run JSON file
    save_dir = preset["path"].get("save_dir", "output")
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(save_dir, f"result_{var_model}_{var_task}_r{var_repeat}_{timestamp}.json")

    with open(out_path, "w") as f:
        json.dump(result, f, indent=4, cls=NumpyEncoder)

    print(f"Results saved to: {out_path}")

    # Also write a human-readable summary alongside the JSON
    formatted = format_result(var_model, var_task, result)
    txt_path = out_path.replace(".json", ".txt")
    with open(txt_path, "w") as f:
        f.write(formatted + "\n")

    print(formatted)

if __name__ == "__main__":
    #
    ##
    start_time = time.time()
    run()
    print("Total time: %s seconds" % (time.time() - start_time))