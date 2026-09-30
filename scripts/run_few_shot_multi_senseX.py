"""
[file]          run_few_shot_multi_senseX.py
[description]   Few-shot knowledge distillation runner for MultiSenseX.

                Trains on a single environment (--env) and tests on every other environment listed
                in preset["data"]["environment"]. The teacher MultiSenseX is trained on the full
                training environment; the student MultiSenseX is trained on a small fraction
                (--few_shot_ratio) of that same environment while being distilled from the frozen
                teacher. MultiSenseX predicts activity and location jointly, so both metric sets are
                reported per test environment.
"""
#
##

import argparse
import random
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import *
from src.data.load_data import load_data_x, load_data_y, encode_data_y
from src.utils import *
from configs.preset import preset

import time
import json
import gc
from datetime import datetime
from pathlib import Path

#
##
def master_splitter(preset, var_model, var_users, var_env="empty_room"):
    """
    [description]
    : build the training split (var_env only) and the test splits (every other environment in
      preset["data"]["environment"]). Labels are the joint activity/location targets used by
      MultiSenseX: activity (N, num_obj_queries, 9) and location (N, num_obj_queries).
    [return]
    : data_train_x, data_train_y_act, data_train_y_loc, test_sets_by_env
      where test_sets_by_env is {env_name: (X, y_act, y_loc)}
    """
    def build(var_environment):
        data_pd_y = load_data_y(preset["path"]["data_y"],
                                var_environment=[var_environment],
                                var_wifi_band=preset["data"]["wifi_band"],
                                var_num_users=var_users)
        var_label_list = data_pd_y["label"].to_list()
        X = load_data_x(preset["path"]["data_x"], var_label_list)
        y_activity = encode_data_y(data_pd_y, "activity")
        y_location = encode_data_y(data_pd_y, "location")
        if var_model == "multiSense_X":
            y_act, y_loc = reduce_dataset_joint_multiSenseX(y_activity, y_location)
        else:
            y_act, y_loc = reduce_dataset_joint(y_activity, y_location, preset["nn"]["num_obj_queries"])
        return X, y_act, y_loc

    data_train_x, data_train_y_act, data_train_y_loc = build(var_env)

    test_sets_by_env = {}
    other_envs = [e for e in preset["data"]["environment"] if e != var_env]
    for e in other_envs:
        X_test, y_act_test, y_loc_test = build(e)
        test_sets_by_env[e] = (X_test, y_act_test, y_loc_test)
        del X_test, y_act_test, y_loc_test
        gc.collect()

    return data_train_x, data_train_y_act, data_train_y_loc, test_sets_by_env


def parse_args():
    """
    [description]
    : parse arguments from input
    """
    var_args = argparse.ArgumentParser()
    var_args.add_argument("--model", default="multiSense_X", type=str)
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
    return var_args.parse_args()


def format_result(result, var_few_shot_ratio, var_kd_weight, var_env, var_epochs, var_teacher_epochs):
    """
    [description]
    : build a formatted string of the per-environment results (activity and location)
    """
    lines = []
    lines.append("=" * 80)
    lines.append("EXPERIMENT RESULTS - Model: MultiSenseX (joint activity + location)")
    lines.append(f"Train env: {var_env} | Few-shot ratio: {var_few_shot_ratio} | KD weight: {var_kd_weight}")
    lines.append(f"Epochs: student {var_epochs} | teacher {var_teacher_epochs}")
    lines.append("=" * 80)
    lines.append("PER_ENV_RESULTS:")
    for env_name in sorted(k for k in result.keys()
                           if isinstance(result[k], dict) and "act" in result[k] and "loc" in result[k]):
        lines.append(f"\n{env_name.upper()}:")
        for key, label in (("act", "ACTIVITY"), ("loc", "LOCATION")):
            stats = result[env_name][key]
            lines.append(f"  {label}:")
            for metric, metric_label in (("PPP", "Perfect Prediction %"), ("accuracy", "Accuracy"),
                                         ("precision", "Precision"), ("recall", "Recall"),
                                         ("f1_score", "F1 Score"), ("total_error", "Total Error")):
                lines.append(f"    Avg {metric_label}: {stats[f'avg_{metric}']:.4f} "
                             f"± {stats[f'se_{metric}']:.4f} (SE)")
    return "\n".join(lines)


def save_result(var_repeat, result):
    """
    [description]
    : save the full result dict to one JSON file per run
    """
    result["model"] = "multiSense_X"
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
        f"result_multiSenseX_fewshot_r{var_repeat}_{timestamp}.json",
    )

    with open(out_path, "w") as f:
        json.dump(result, f, indent=4, cls=NumpyEncoder)

    return out_path


#
##
def run():
    """
    [description]
    : run few-shot knowledge distillation for MultiSenseX (train on one env, test on the others)
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
    var_model = var_args.model
    var_repeat = var_args.repeat
    var_users = [u.strip() for u in var_args.users.split(',')]
    var_env = var_args.env
    var_teacher_epochs = var_args.teacher_epochs if var_args.teacher_epochs is not None else var_args.epochs

    # Ensuring there is no data leakage while doing splits.
    (data_train_x, data_train_y_act, data_train_y_loc,
     test_sets_by_env) = master_splitter(preset, var_model, var_users, var_env)

    save_path = Path(f'./visualizations/few_shot_multi_senseX/{var_env}/1')
    while save_path.is_dir():
        new_name = str((int(save_path.name) + 1))
        save_path = save_path.parent / new_name

    #
    ## run few-shot distillation
    result = run_multi_senseX_few_shot(
        data_train_x, data_train_y_act, data_train_y_loc,
        test_sets_by_env,
        var_few_shot_ratio=var_args.few_shot_ratio,
        var_kd_weight=var_args.kd_weight,
        var_kd_temperature=var_args.kd_temperature,
        var_teacher_epochs=var_teacher_epochs,
        var_student_epochs=var_args.epochs,
        var_repeat=var_repeat, var_env=var_env, save_path=save_path)

    #
    ##
    result["few_shot_ratio"] = var_args.few_shot_ratio
    result["kd_weight"] = var_args.kd_weight
    result["kd_temperature"] = var_args.kd_temperature
    result["train_env"] = var_env
    result["student_epochs"] = var_args.epochs
    result["teacher_epochs"] = var_teacher_epochs

    formatted = format_result(result, var_args.few_shot_ratio, var_args.kd_weight,
                              var_env, var_args.epochs, var_teacher_epochs)
    out_path = save_result(var_repeat, result)

    print(formatted)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":

    start_time = time.time()
    run()
    print("Total time: %s seconds" % (time.time() - start_time))
