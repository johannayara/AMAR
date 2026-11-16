"""
[file]          reformat_run_report.py
[description]   Rewrite the human-readable report (.txt) of an already saved few-shot run from its
                .json, so a report written by an older/broken formatter is corrected without
                re-running the experiment. The formatting comes from the runner's own format_result,
                imported by path, so the two can never drift.

                    python scripts/test_scripts/reformat_run_report.py output/few_shot_multi_rooms/RUN.json
                    python scripts/test_scripts/reformat_run_report.py output/few_shot_multi_rooms

                Given a directory, every *.json directly under it is rewritten next to its .txt.
"""
#
##

import argparse
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

#
##


def parse_args():
    """
    [description]
    : command line arguments
    """
    var_args = argparse.ArgumentParser()
    var_args.add_argument("paths", nargs="+", help="run .json files, or directories holding them")
    return var_args.parse_args()


def load_formatter():
    """
    [description]
    : import format_result from scripts/run_few_shot_multi_rooms.py by path. Importing the module runs
      its imports only -- the runner itself is behind an "if __name__ == '__main__'" guard.
    """
    var_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'run_few_shot_multi_rooms.py'))
    var_spec = importlib.util.spec_from_file_location("run_few_shot_multi_rooms", var_path)
    var_module = importlib.util.module_from_spec(var_spec)
    var_spec.loader.exec_module(var_module)
    return var_module.format_result


def find_jsons(var_paths):
    """
    [description]
    : expand the arguments to run .json files: a directory contributes its direct *.json children.
    """
    var_out = []
    for var_path in var_paths:
        if os.path.isdir(var_path):
            var_out.extend(os.path.join(var_path, var_name) for var_name in sorted(os.listdir(var_path))
                           if var_name.endswith(".json"))
        else:
            var_out.append(var_path)
    return var_out


def reformat(var_json_path, var_format_result):
    """
    [description]
    : rewrite <stem>.txt from <stem>.json using the runner's formatter.
    """
    with open(var_json_path) as var_file:
        var_result = json.load(var_file)
    if "per_env" not in var_result:
        print(f"skip {var_json_path}: no 'per_env' key")
        return
    var_formatted = var_format_result(var_result.get("model", "-"), var_result.get("task", "-"),
                                      var_result["per_env"],
                                      var_result.get("few_shot_ratio", float("nan")),
                                      var_result.get("kd_weight", float("nan")),
                                      "+".join(var_result.get("train_rooms", [])))
    var_txt_path = os.path.splitext(var_json_path)[0] + ".txt"
    with open(var_txt_path, "w") as var_file:
        var_file.write(var_formatted + "\n")
    print(f"rewrote {var_txt_path}")


if __name__ == "__main__":
    #
    var_args = parse_args()
    var_format_result = load_formatter()
    for var_json_path in find_jsons(var_args.paths):
        reformat(var_json_path, var_format_result)
