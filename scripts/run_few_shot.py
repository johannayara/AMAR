"""
[file]          run_few_shot.py
[description]   Few-shot knowledge distillation runner for AMAR_WO_RVQ and density_map.

                Trains on a single environment (--env) and tests on every other environment listed
                in preset["data"]["environment"]. The teacher is trained on the full training
                environment; the student is trained on a small fraction (--few_shot_ratio) of that
                same environment while being distilled from the frozen teacher.
"""
#
##

import argparse
import random
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import *
from src.data.load_data import load_data_x, load_data_y, encode_data_y, encode_occupancy_y
from src.utils import *
from configs.preset import preset

import time
import json
import gc
from datetime import datetime
from pathlib import Path

#
##
def master_splitter(preset, var_task, var_model, var_users, var_env="empty_room"):
    """
    [description]
    : build the training split (var_env only) and the test splits (every other environment in
      preset["data"]["environment"]).
    [return]
    : data_train_x, data_train_y, test_sets_by_env (dict {env_name: (X, y)})
    """
    data_pd_y = load_data_y(preset["path"]["data_y"],
                            var_environment=[var_env],
                            var_wifi_band=preset["data"]["wifi_band"],
                            var_num_users=var_users)
    var_label_list = data_pd_y["label"].to_list()
    data_train_x = load_data_x(preset["path"]["data_x"], var_label_list)
    if var_model == "density_map":
        ## Density-map group counting predicts per-location occupancy, so the label is the room's
        ## occupancy vector rather than a task encoding.
        data_train_y = encode_occupancy_y(data_pd_y, var_env)
    else:
        data_train_y = encode_data_y(data_pd_y, var_task)
        if var_model in ("AMAR_WO_RVQ", "AMAR"):
            data_train_y = reduce_dataset(data_train_y, var_task, preset["nn"]["num_obj_queries"])

    test_sets_by_env = {}
    other_envs = [e for e in preset["data"]["environment"] if e != var_env]
    for e in other_envs:
        data_pd_y = load_data_y(preset["path"]["data_y"],
                                var_environment=[e],
                                var_wifi_band=preset["data"]["wifi_band"],
                                var_num_users=var_users)
        var_label_list = data_pd_y["label"].to_list()
        X_test = load_data_x(preset["path"]["data_x"], var_label_list)
        if var_model == "density_map":
            y_test = encode_occupancy_y(data_pd_y, e)
        else:
            y_test = encode_data_y(data_pd_y, var_task)
            if var_model in ("AMAR_WO_RVQ", "AMAR"):
                y_test = reduce_dataset(y_test, var_task, preset["nn"]["num_obj_queries"])
        test_sets_by_env[e] = (X_test, y_test)
        del X_test, y_test
        gc.collect()

    return data_train_x, data_train_y, test_sets_by_env


def parse_args():
    """
    [description]
    : parse arguments from input
    """
    var_args = argparse.ArgumentParser()
    var_args.add_argument("--model", default=preset["model"], type=str)
    var_args.add_argument("--task", default=preset["task"], type=str)
    var_args.add_argument("--repeat", default=preset["repeat"], type=int)
    var_args.add_argument("--users", default="0,1,2,3,4,5", type=str, help="Comma-separated list of user IDs")
    var_args.add_argument("--env", default="empty_room", type=str, help="the single training room")
    var_args.add_argument("--few_shot_ratio", default=0.05, type=float,
                          help="fraction of the training environment used to train the student")
    var_args.add_argument("--epochs", default=200, type=int,
                          help="student training epochs (default 200; the student only sees a few-shot "
                               "slice, so 20 epochs is a handful of optimizer steps)")
    var_args.add_argument("--teacher_epochs", default=None, type=int,
                          help="teacher training epochs (defaults to --epochs)")
    var_args.add_argument("--kd_weight", default=1.0, type=float, help="weight of the distillation loss")
    var_args.add_argument("--kd_temperature", default=1.0, type=float, help="temperature of the soft targets")
    var_args.add_argument("--no_compile", action="store_true", help="disable torch.compile of the feature extractors")
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
            lines.append(f"  Avg Count MAE: {stats['avg_mae']:.4f} ± {stats['se_mae']:.4f} (SE)")
            lines.append(f"  Avg Occupancy Accuracy: {stats['avg_occupancy_accuracy']:.4f} "
                         f"± {stats['se_occupancy_accuracy']:.4f} (SE)")
            lines.append(f"  Avg Occupancy F1: {stats['avg_occupancy_f1']:.4f} "
                         f"± {stats['se_occupancy_f1']:.4f} (SE)")
            if "avg_loc_error" in stats:
                lines.append(f"  Avg Loc Error: {stats['avg_loc_error']:.4f} "
                             f"± {stats['se_loc_error']:.4f} (SE)")
                lines.append(f"  Avg Loc Detection: {stats['avg_loc_detection']:.4f} "
                             f"± {stats['se_loc_detection']:.4f} (SE)")
        else:
            for metric, label in (("precision", "Precision"), ("recall", "Recall"),
                                  ("PPP", "Perfect Prediction %"), ("f1_score", "F1 Score"),
                                  ("accuracy", "Accuracy"), ("total_error", "Total Error")):
                if f"avg_{metric}" in stats:
                    lines.append(f"  Avg {label}: {stats[f'avg_{metric}']:.4f} "
                                 f"± {stats[f'se_{metric}']:.4f} (SE)")
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
    os.makedirs(os.path.join(save_dir, "json"), exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        save_dir,
        "json",
        f"result_{var_model}_fewshot_{var_task}_r{var_repeat}_{timestamp}.json",
    )

    with open(out_path, "w") as f:
        json.dump(result, f, indent=4, cls=NumpyEncoder)

    return out_path


#
##
def run():
    """
    [description]
    : run few-shot knowledge distillation (train on one env, test on the others). Dispatches to
      run_density_map_few_shot for the density_map model and run_AMAR_WO_RVQ_few_shot otherwise.
    """
    SEED = 103  # Ensuring the results are reproducible
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
    var_teacher_epochs = var_args.teacher_epochs
    if var_teacher_epochs is None:
        var_teacher_epochs = var_args.epochs

    # Ensuring there is no data leakage while doing splits.
    data_train_x, data_train_y, test_sets_by_env = master_splitter(
        preset, var_task, var_model, var_users, var_env)

    save_path = Path(f'./visualizations/few_shot/{var_env}/{var_task}/1')
    while save_path.is_dir():
        new_name = str((int(save_path.name) + 1))
        save_path = save_path.parent / new_name

    #
    ## run few-shot distillation
    if var_model == "density_map":
        result = run_density_map_few_shot(
            data_train_x, data_train_y,
            test_sets_by_env,
            var_few_shot_ratio=var_args.few_shot_ratio,
            var_kd_weight=var_args.kd_weight,
            var_kd_temperature=var_args.kd_temperature,
            var_teacher_epochs=var_teacher_epochs,
            var_student_epochs=var_args.epochs,
            var_compile=not var_args.no_compile,
            var_repeat=var_repeat, var_task=var_task, var_env=var_env, save_path=save_path)
    else:
        result = run_AMAR_WO_RVQ_few_shot(
            data_train_x, data_train_y,
            test_sets_by_env,
            var_few_shot_ratio=var_args.few_shot_ratio,
            var_kd_weight=var_args.kd_weight,
            var_kd_temperature=var_args.kd_temperature,
            var_teacher_epochs=var_teacher_epochs,
            var_student_epochs=var_args.epochs,
            var_compile=not var_args.no_compile,
            var_repeat=var_repeat, var_task=var_task, var_env=var_env, save_path=save_path)

    #
    ##
    result["model"] = var_model
    result["task"] = var_task
    result["repeat"] = var_repeat
    result["data"] = preset["data"]
    result["nn"] = preset["nn"]
    result["few_shot_ratio"] = var_args.few_shot_ratio
    result["kd_weight"] = var_args.kd_weight
    result["kd_temperature"] = var_args.kd_temperature
    result["train_env"] = var_env
    result["student_epochs"] = var_args.epochs
    result["teacher_epochs"] = var_teacher_epochs

    formatted = format_result(var_model, var_task, result, var_args.few_shot_ratio, var_args.kd_weight, var_env)
    out_path = save_result(var_model, var_task, var_repeat, result)

    print(formatted)
    print(f"\nResults saved to: {out_path}")

    # Release the multi-GB per-task arrays before the next task
    del data_train_x, data_train_y, test_sets_by_env
    gc.collect()


if __name__ == "__main__":

    start_time = time.time()
    run()
    print("Total time: %s seconds" % (time.time() - start_time))
