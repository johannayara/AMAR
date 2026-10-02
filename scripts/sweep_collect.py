"""
[file]          sweep_collect.py
[description]   Collect the per-run JSON files written by scripts/run_main.py during the
                training-set-size sweep and print one row per (environment, training-set size).

                Usage:  python scripts/sweep_collect.py [--dir output/json] [--metric avg_f1_score]
"""
#
##

import argparse
import glob
import json
import os

METRICS = ["avg_accuracy", "avg_PPP", "avg_precision", "avg_recall", "avg_f1_score", "avg_total_error"]


def parse_args():
    var_args = argparse.ArgumentParser()
    var_args.add_argument("--dir", default=os.path.join("output", "json"), type=str)
    var_args.add_argument("--model", default=None, type=str, help="filter by model name")
    var_args.add_argument("--metric", default="avg_f1_score", type=str,
                          help="metric used to report the fraction of the full-data performance")
    return var_args.parse_args()


def load_rows(var_dir, var_model):
    var_rows = []
    for var_path in glob.glob(os.path.join(var_dir, "result_*.json")):
        with open(var_path) as var_file:
            var_result = json.load(var_file)
        if var_model and var_result.get("model") != var_model:
            continue
        #
        ## layered models (AMAR) report per decoder layer; the last layer is the trained output
        var_layer_keys = sorted(k for k in var_result if isinstance(k, str) and k.startswith("layer_"))
        var_metrics = var_result[var_layer_keys[-1]] if var_layer_keys else var_result
        #
        var_samples = var_result.get("train_samples", 0)
        var_rows.append({
            "model": var_result.get("model", "?"),
            "task": var_result.get("task", "?"),
            "env": var_result.get("env", "?"),
            "n": "all" if not var_samples else str(var_samples),
            "repeat": var_result.get("repeat", "?"),
            "path": var_path,
            **{var_metric: var_metrics.get(var_metric) for var_metric in METRICS},
        })
    return var_rows


def main():
    var_args = parse_args()
    var_rows = load_rows(var_args.dir, var_args.model)
    if not var_rows:
        print(f"No result_*.json found under {var_args.dir}. Run the sweep first.")
        return
    #
    ## Reference = full training set (n=all), per (model, task, env)
    var_reference = {(r["model"], r["task"], r["env"]): r for r in var_rows if r["n"] == "all"}
    #
    var_header = ["model", "task", "env", "n"] + METRICS + [f"{var_args.metric}/all"]
    print("\t".join(var_header))
    for var_row in sorted(var_rows, key=lambda r: (r["model"], r["task"], r["env"], r["n"] == "all", int(r["n"]) if r["n"] != "all" else 10**9)):
        var_ref = var_reference.get((var_row["model"], var_row["task"], var_row["env"]))
        var_ratio = "-"
        if var_ref and var_ref.get(var_args.metric) not in (None, 0) and var_row.get(var_args.metric) is not None:
            var_ratio = f"{var_row[var_args.metric] / var_ref[var_args.metric]:.3f}"
        var_cells = [var_row["model"], var_row["task"], var_row["env"], var_row["n"]]
        var_cells += [f"{var_row[var_metric]:.4f}" if var_row.get(var_metric) is not None else "-"
                      for var_metric in METRICS]
        var_cells.append(var_ratio)
        print("\t".join(var_cells))


if __name__ == "__main__":
    main()
