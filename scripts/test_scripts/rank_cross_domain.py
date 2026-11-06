#!/usr/bin/env python3
"""
[file]          rank_cross_domain.py
[description]   Answer "which model is best at cross-domain localization within the WiMANS rooms?"
                from what is on disk.

                It reuses ablation_report's parsers (one source of truth for reading a report block) and
                pools every configuration across the leave-one-room-out folds, so a config is scored on
                all three held-out rooms rather than one. For each (protocol, model, head, backbone,
                encoder_norm) it prints the fold count, the pooled mean of the primary metric and its
                standard error, and the per-fold values.

                The primary metric is chosen by what the record actually reports:
                    position_error_m   density / coords heads      lower is better
                    f1 | micro_f1      AMAR_WO_RVQ (set classes)  higher is better
                A configuration with fewer than the full fold count is still printed (n shows how many),
                so a matrix that only ran two of three folds is visible rather than silently ranked.

                The dummy_model rows are the training-prior baseline: a config whose cross-domain metric
                does not beat the dummy has not learned a transferable mapping. They are printed in a
                separate block, not mixed into the ranking (the dummy reports a classification metric,
                the heads a position error).

                Usage:
                    python scripts/test_scripts/rank_cross_domain.py
                    python scripts/test_scripts/rank_cross_domain.py --protocols lor_matrix
                    python scripts/test_scripts/rank_cross_domain.py --include-archived
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ablation_report as ar   # noqa: E402  (sibling script, needs the path insert above)

## metric -> (lower_is_better). The first key a record carries is the one it is ranked on.
PRIMARY = {"position_error_m": True, "f1": False, "micro_f1": False, "present_f1": False}


def parse_args():
    var_args = argparse.ArgumentParser()
    var_args.add_argument("--output-dir", default="output", type=str)
    var_args.add_argument("--protocols", default="lor_matrix,cross_domain", type=str,
                          help="comma-separated protocols to rank (lor_matrix=leave-one-room-out, "
                               "cross_domain=standalone runs, matrix=single-room)")
    var_args.add_argument("--include-archived", action="store_true",
                          help="also read old_results/output (off by default: those predate the "
                               "coords-decode / gate-consistent fixes)")
    var_args.add_argument("--json", default=None, type=str, help="dump the pooled rows as JSON")
    return var_args.parse_args()


def _fold(var_rec):
    return var_rec.holdout or (var_rec.test_envs[0] if var_rec.test_envs else None)


def _metric_for(var_rec):
    """(metric, value, fold) of a record, or (None, None, None) when it reports nothing rankable."""
    var_fold = _fold(var_rec)
    if not var_fold:
        return None, None, None
    var_metrics = var_rec.metrics.get(var_fold) or next(iter(var_rec.metrics.values()), {})
    for var_metric in PRIMARY:
        if var_metric in var_metrics and var_metrics[var_metric] is not None:
            return var_metric, var_metrics[var_metric], var_fold
    return None, None, None


def _pooled(var_values):
    var_n = len(var_values)
    var_mean = sum(var_values) / var_n
    if var_n > 1:
        var_var = sum((var_value - var_mean) ** 2 for var_value in var_values) / (var_n - 1)
        var_se = (var_var / var_n) ** 0.5
    else:
        var_se = 0.0
    return var_mean, var_se


def main():
    var_args = parse_args()
    var_protocols = {var_protocol.strip() for var_protocol in var_args.protocols.split(",")}
    var_roots = [var_args.output_dir]
    if var_args.include_archived and os.path.isdir(os.path.join("old_results", "output")):
        var_roots.append(os.path.join("old_results", "output"))
    var_records = ar.collect(var_roots)

    var_groups = defaultdict(lambda: defaultdict(dict))   # key -> metric -> {fold: value}
    var_dummy = defaultdict(lambda: defaultdict(dict))
    for var_rec in var_records:
        if var_rec.protocol not in var_protocols:
            continue
        var_metric, var_value, var_fold = _metric_for(var_rec)
        if var_metric is None:
            continue
        var_key = (var_rec.protocol, var_rec.model, var_rec.head, var_rec.backbone, var_rec.encoder_norm)
        if var_rec.model == "dummy_model":
            var_dummy[var_key][var_metric][var_fold] = var_value
        else:
            var_groups[var_key][var_metric][var_fold] = var_value

    var_rows = []
    for var_key, var_by_metric in var_groups.items():
        for var_metric, var_folds in var_by_metric.items():
            var_values = list(var_folds.values())
            var_mean, var_se = _pooled(var_values)
            var_rows.append({
                "protocol": var_key[0], "model": var_key[1], "head": var_key[2] or "-",
                "backbone": var_key[3] or "-", "encoder_norm": var_key[4] or "-",
                "metric": var_metric, "n": len(var_values), "mean": var_mean, "se": var_se,
                "folds": dict(sorted(var_folds.items())),
            })
    ## lower-is-better metrics ascending, higher-is-better descending; then by n, so a config that ran
    ## every fold outranks one that ran a subset at the same mean
    var_rows.sort(key=lambda var_row: ((var_row["mean"] if PRIMARY[var_row["metric"]] else -var_row["mean"]),
                                       -var_row["n"]))

    print("=" * 100)
    print(f"CROSS-DOMAIN WiMANS RANKING - protocols {sorted(var_protocols)}, "
          f"{'including' if var_args.include_archived else 'live output only'}")
    print("=" * 100)
    print(f"{'model':13s} {'head':7s} {'backbone':14s} {'norm':6s} {'metric':16s} {'n':>2s} "
          f"{'mean':>9s} {'se':>8s}  folds")
    print("-" * 100)
    for var_row in var_rows:
        var_folds = " ".join(f"{var_fold}={var_value:.4f}" for var_fold, var_value in var_row["folds"].items())
        print(f"{str(var_row['model']):13s} {var_row['head']:7s} {var_row['backbone']:14s} "
              f"{var_row['encoder_norm']:6s} {var_row['metric']:16s} {var_row['n']:2d} "
              f"{var_row['mean']:9.4f} {var_row['se']:8.4f}  {var_folds}")

    print("\nTRAINING-PRIOR BASELINE (dummy_model; a real config must beat it)")
    print(f"{'model':13s} {'head':7s} {'backbone':14s} {'norm':6s} {'metric':16s} {'n':>2s} "
          f"{'mean':>9s} {'se':>8s}")
    print("-" * 100)
    for var_key, var_by_metric in sorted(var_dummy.items()):
        for var_metric, var_folds in var_by_metric.items():
            var_mean, var_se = _pooled(list(var_folds.values()))
            print(f"{str(var_key[1]):13s} {str(var_key[2] or '-'):7s} {str(var_key[3] or '-'):14s} "
                  f"{str(var_key[4] or '-'):6s} {var_metric:16s} {len(var_folds):2d} "
                  f"{var_mean:9.4f} {var_se:8.4f}")

    if var_args.json:
        import json
        with open(var_args.json, "w") as var_file:
            json.dump(var_rows, var_file, indent=2)
        print(f"\nrows written to {var_args.json}")


if __name__ == "__main__":
    main()
