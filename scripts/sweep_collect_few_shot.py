"""
[file]          sweep_collect_few_shot.py
[description]   Rank the runs written by scripts/sweep_few_shot_hwild.sh.

                Each combo dir under <dir> holds one run's JSON (scripts/run_few_shot_hwild_to_wimans.py
                output). The JSON is self-describing, so the settings are read back from it rather than
                from the directory name: kd_weight, few_shot_ratio, init_from_teacher, standardize,
                window, stride, repeat. The score is the mean student MDE over the query rooms -- the
                same quantity the report prints -- next to the teacher and constant baselines, so a
                config that only looks good by beating a weak teacher is visible.

                Usage:
                    python scripts/sweep_collect_few_shot.py
                    python scripts/sweep_collect_few_shot.py --metric avg_student_rmse_m --top 10
"""
#
##

import argparse
import csv
import glob
import json
import math
import os

## the three student query metrics the runner reports, plus their teacher/constant counterparts
STUDENT_METRICS = ("avg_student_mean_error_m", "avg_student_rmse_m", "avg_student_median_error_m")


def parse_args():
    var_args = argparse.ArgumentParser()
    var_args.add_argument("--dir", default=os.path.join("output", "sweep", "few_shot_hwild_wimans"),
                          type=str, help="sweep root written by scripts/sweep_few_shot_hwild.sh")
    var_args.add_argument("--metric", default="avg_student_mean_error_m", type=str, choices=STUDENT_METRICS,
                          help="metric to rank on (minimized), averaged over the query rooms")
    var_args.add_argument("--top", default=25, type=int, help="rows to print (0 = all)")
    var_args.add_argument("--csv", default=None, type=str,
                          help="CSV path (default <dir>/summary.csv; empty string disables)")
    return var_args.parse_args()


def _pooled_mean(var_values):
    """
    [description]
    : mean of a list of per-room values, and the pooled standard error sqrt(sum(se^2))/n.
    : var_values: list of (value, se) tuples, one per room
    : return: (mean, pooled_se) or (None, None) if empty
    """
    var_values = [(var_value, var_se) for var_value, var_se in var_values if var_value is not None]
    if not var_values:
        return None, None
    var_mean = sum(var_value for var_value, _ in var_values) / len(var_values)
    var_se = math.sqrt(sum((var_se or 0.0) ** 2 for _, var_se in var_values)) / len(var_values)
    return var_mean, var_se


def load_rows(var_dir):
    """
    [description]
    : one row per combo directory, from the newest JSON in it (a re-run leaves the older one behind).
    : return: list of dicts
    """
    var_rows = []
    for var_combo_dir in sorted(glob.glob(os.path.join(var_dir, "*"))):
        if not os.path.isdir(var_combo_dir):
            continue
        var_jsons = glob.glob(os.path.join(var_combo_dir, "*.json"))
        if not var_jsons:
            continue
        var_path = max(var_jsons, key=os.path.getmtime)
        with open(var_path) as var_file:
            var_result = json.load(var_file)
        var_per_env = var_result.get("per_env") or {}
        if not var_per_env:
            continue
        #
        ## aggregate the student metric (the score) and the teacher/constant baselines over the rooms
        var_student = {var_name: _pooled_mean(
            [(var_env.get(var_name), var_env.get("se_student_" + var_name[len("avg_student_"):]))
             for var_env in var_per_env.values()]) for var_name in STUDENT_METRICS}
        var_teacher_mean, _ = _pooled_mean(
            [(var_env.get("avg_teacher_mean_error_m"), var_env.get("se_teacher_mean_error_m"))
             for var_env in var_per_env.values()])
        var_const_mean, _ = _pooled_mean(
            [(var_env.get("avg_const_mean_error_m"), None) for var_env in var_per_env.values()])
        #
        var_row = {
            "tag": os.path.basename(var_combo_dir),
            "kd_weight": var_result.get("kd_weight"),
            "few_shot_ratio": var_result.get("few_shot_ratio"),
            "init_from_teacher": var_result.get("init_from_teacher"),
            "standardize": var_result.get("standardize"),
            "window": var_result.get("window"),
            "stride": var_result.get("stride"),
            "repeat": var_result.get("repeat"),
            "rooms": len(var_per_env),
            "teacher_mde": var_teacher_mean,
            "const_mde": var_const_mean,
            "json": var_path,
        }
        for var_name in STUDENT_METRICS:
            var_row[var_name], var_row["se_" + var_name] = var_student[var_name]
        var_rows.append(var_row)
    return var_rows


def _fmt(var_value, var_places=4):
    return f"{var_value:.{var_places}f}" if isinstance(var_value, (int, float)) and math.isfinite(var_value) else "-"


def main():
    var_args = parse_args()
    var_rows = load_rows(var_args.dir)
    if not var_rows:
        print(f"No per-combo JSON found under {var_args.dir}. Run scripts/sweep_few_shot_hwild.sh first.")
        return
    #
    var_metric = var_args.metric
    var_scored = [var_row for var_row in var_rows if var_row.get(var_metric) is not None]
    if not var_scored:
        print(f"None of the {len(var_rows)} combos reports {var_metric}.")
        return
    var_scored.sort(key=lambda var_row: var_row[var_metric])
    #
    var_header = ["tag", "kd", "fs", "init", "std", "win", "stride", "rep",
                  "student_MDE", "se", "teacher_MDE", "const_MDE", "vs_const", "vs_teacher", "beats_const"]
    print("\t".join(var_header))
    for var_row in (var_scored if var_args.top <= 0 else var_scored[:var_args.top]):
        var_vs_const = (var_row["const_mde"] - var_row[var_metric]
                        if var_row["const_mde"] is not None and var_row[var_metric] is not None else None)
        var_vs_teacher = (var_row["teacher_mde"] - var_row[var_metric]
                          if var_row["teacher_mde"] is not None and var_row[var_metric] is not None else None)
        var_cells = [
            var_row["tag"], str(var_row["kd_weight"]), str(var_row["few_shot_ratio"]),
            "teacher" if var_row["init_from_teacher"] else "random", str(var_row["standardize"]),
            str(var_row["window"]), str(var_row["stride"]), str(var_row["repeat"]),
            _fmt(var_row[var_metric]), _fmt(var_row["se_" + var_metric]),
            _fmt(var_row["teacher_mde"]), _fmt(var_row["const_mde"]),
            _fmt(var_vs_const), _fmt(var_vs_teacher),
            "yes" if (var_vs_const is not None and var_vs_const > 0) else "no",
        ]
        print("\t".join(var_cells))
    #
    ## the winner: the combo that beats the constant baseline by the most, or the raw best when none do
    var_winners = [var_row for var_row in var_scored
                   if var_row["const_mde"] is not None and var_row[var_metric] < var_row["const_mde"]]
    var_best = (max(var_winners, key=lambda var_row: var_row["const_mde"] - var_row[var_metric])
                if var_winners else var_scored[0])
    print("\nBest config" + ("" if var_winners else " (none beats the constant baseline)"))
    print(f"  tag               : {var_best['tag']}")
    print(f"  {var_metric:<18}: {_fmt(var_best[var_metric])} m (SE {_fmt(var_best['se_' + var_metric])}) "
          f"over {var_best['rooms']} rooms, {var_best['repeat']} repeats")
    print(f"  teacher / constant: {_fmt(var_best['teacher_mde'])} m / {_fmt(var_best['const_mde'])} m")
    print(f"  json              : {var_best['json']}")
    #
    var_csv = var_args.csv if var_args.csv is not None else os.path.join(var_args.dir, "summary.csv")
    if var_csv:
        with open(var_csv, "w", newline="") as var_file:
            var_writer = csv.DictWriter(var_file, fieldnames=list(var_scored[0].keys()))
            var_writer.writeheader()
            var_writer.writerows(var_scored)
        print(f"\nAll {len(var_scored)} combos written to {var_csv}")


if __name__ == "__main__":
    main()
