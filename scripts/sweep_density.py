"""
[file]          sweep_density.py
[description]   Successive-halving hyperparameter search for the density-map model in a single room.

                Each trial is one `scripts/run_main.py` invocation with `--hp` overrides, so trials
                are isolated processes and a crashed trial cannot poison the others. The three stages
                trade breadth for depth: many configs at a few epochs, then the survivors at full
                length with repeats.

                Protocol notes:
                  - The data split is the deterministic one in run_main.master_splitter (random_state
                    103) and run_density_map (random_state 39), so every trial sees the same samples.
                  - `density.eval_threshold=0.5` is pinned by default. Calibrating the threshold on the
                    split that is then scored is optimistic and lets a config win by fitting the
                    threshold sweep instead of the model. Pass --calibrated_threshold to opt out.
                  - Rank on per-location occupancy F1 (dense signal), tie-break on count MAE. Exact
                    count accuracy is a coarse, threshold-sensitive sanity check, not the selector.

                Usage:
                    python scripts/sweep_density.py --env empty_room --out output/sweep/density_map_hp
                    python scripts/sweep_density.py --env empty_room --workers 4 --gpus 0,1 --dry_run
"""
#
##

import argparse
import concurrent.futures
import glob
import itertools
import json
import os
import subprocess
import sys

import numpy as np

#
## dotted preset key -> sampler. Only these keys are searched; everything else stays at preset.
SEARCH_SPACE = {
    "nn.lr": ("loguniform", 1e-4, 2e-3),
    "nn.batch_size": ("choice", [8, 16, 32]),
    "nn.weight_decay": ("loguniform", 1e-5, 1e-3),
    "density.decoder_hidden": ("choice", [32, 64, 128]),
    "density.decoder_dropout": ("choice", [0.0, 0.1, 0.3]),
    "density.balance_empty_class": ("choice", [True, False]),
    "density.sigma": ("choice", [0.04, 0.06, 0.08]),
    "nn.scheduler.num_warmup_epochs": ("choice", [1, 3, 5]),
    "nn.scheduler.min_lr_ratio": ("choice", [0.01, 0.1]),
}


def parse_args():
    """
    [description]
    : parse arguments from input
    """
    var_args = argparse.ArgumentParser()
    var_args.add_argument("--env", default="empty_room", type=str, help="the single training room")
    var_args.add_argument("--out", default=os.path.join("output", "sweep", "density_map_hp"), type=str)
    var_args.add_argument("--seed", default=39, type=int, help="random-search seed")
    var_args.add_argument("--metric", default="avg_occupancy_f1", type=str,
                          help="metric to rank on (max)")
    var_args.add_argument("--tie_break", default="avg_mae", type=str,
                          help="metric to break ties on (min)")
    var_args.add_argument("--calibrated_threshold", action="store_true",
                          help="let each trial calibrate its threshold on the validation split instead "
                               "of pinning density.eval_threshold=0.5")
    var_args.add_argument("--workers", default=1, type=int, help="trials to run in parallel")
    var_args.add_argument("--gpus", default="0", type=str,
                          help="comma-separated GPU ids, assigned round-robin to the workers")
    var_args.add_argument("--dry_run", action="store_true", help="print the trial commands and exit")
    var_args.add_argument("--resume", action="store_true",
                          help="reuse a trial's JSON if it already exists")
    #
    ## stage budgets: (trials, epochs, repeats, promote)
    var_args.add_argument("--stage1_trials", default=24, type=int)
    var_args.add_argument("--stage1_epochs", default=25, type=int)
    var_args.add_argument("--stage2_trials", default=6, type=int)
    var_args.add_argument("--stage2_epochs", default=100, type=int)
    var_args.add_argument("--stage2_repeats", default=3, type=int)
    var_args.add_argument("--stage3_trials", default=2, type=int)
    var_args.add_argument("--stage3_epochs", default=100, type=int)
    var_args.add_argument("--stage3_repeats", default=5, type=int)
    #
    return var_args.parse_args()


def sample_config(var_rng):
    """
    [description]
    : draw one random config from SEARCH_SPACE.
    : return: dict {dotted_key: value}
    """
    var_config = {}
    for var_key, var_spec in SEARCH_SPACE.items():
        if var_spec[0] == "choice":
            var_config[var_key] = var_rng.choice(var_spec[1])
        else:
            var_config[var_key] = float(var_rng.uniform(var_spec[1], var_spec[2]))
    return var_config


def format_hp_value(var_value):
    """
    [description]
    : render a config value for the --hp command line. Numbers/bools keep their literal form; strings
      are passed unquoted (run_main also accepts quotes, but unquoted is unambiguous for paths).
    """
    if isinstance(var_value, bool):
        return "true" if var_value else "false"
    if isinstance(var_value, (int, float)):
        return repr(var_value)
    return str(var_value)


def run_trial(var_args, var_stage, var_trial_idx, var_config):
    """
    [description]
    : run one trial as a run_main.py subprocess and return its result dict (or None on failure).
    """
    var_trial_dir = os.path.join(var_args.out, f"stage{var_stage}_trial{var_trial_idx:04d}")
    var_json_dir = os.path.join(var_trial_dir, "json")
    #
    if var_args.resume:
        var_existing = sorted(glob.glob(os.path.join(var_json_dir, "result_*.json")))
        if var_existing:
            with open(var_existing[-1]) as var_file:
                return json.load(var_file)
    #
    os.makedirs(var_trial_dir, exist_ok=True)
    var_hp = dict(var_config)
    var_hp["path.save_dir"] = var_trial_dir
    if not var_args.calibrated_threshold:
        var_hp["density.eval_threshold"] = 0.5
    #
    var_epochs = var_args.stage_epochs
    var_repeats = var_args.stage_repeats
    var_cmd = [sys.executable, os.path.join("scripts", "run_main.py"),
               "--model", "density_map", "--task", "location",
               "--env", var_args.env, "--repeat", str(var_repeats), "--epochs", str(var_epochs)]
    for var_key, var_value in var_hp.items():
        var_cmd += ["--hp", f"{var_key}={format_hp_value(var_value)}"]
    #
    if var_args.dry_run:
        print("  " + " ".join(var_cmd))
        return None
    #
    var_env_vars = dict(os.environ)
    var_env_vars["WANDB_MODE"] = "offline"
    var_env_vars["CUDA_VISIBLE_DEVICES"] = var_args.gpu_id
    with open(os.path.join(var_trial_dir, "run.txt"), "w") as var_log:
        subprocess.run(var_cmd, stdout=var_log, stderr=subprocess.STDOUT, env=var_env_vars)
    #
    var_paths = sorted(glob.glob(os.path.join(var_json_dir, "result_*.json")))
    if not var_paths:
        print(f"  trial {var_trial_idx}: no result JSON (see {var_trial_dir}/run.txt)")
        return None
    with open(var_paths[-1]) as var_file:
        var_result = json.load(var_file)
    var_result["_trial_dir"] = var_trial_dir
    return var_result


def run_stage(var_args, var_stage, var_configs, var_epochs, var_repeats):
    """
    [description]
    : run every config of one stage, in parallel up to --workers, and return [(config, result)].
    """
    var_args.stage_epochs = var_epochs
    var_args.stage_repeats = var_repeats
    var_gpus = [g.strip() for g in var_args.gpus.split(",") if g.strip()]
    #
    print(f"\n=== stage {var_stage}: {len(var_configs)} trials x {var_epochs} epochs "
          f"x {var_repeats} repeats ===")
    var_out = []
    if var_args.dry_run:
        for var_idx, var_config in enumerate(var_configs):
            var_args.gpu_id = var_gpus[var_idx % len(var_gpus)]
            run_trial(var_args, var_stage, var_idx, var_config)
        return var_out
    #
    def var_worker(var_item):
        var_idx, var_config = var_item
        var_local = argparse.Namespace(**vars(var_args))
        var_local.gpu_id = var_gpus[var_idx % len(var_gpus)]
        return var_config, run_trial(var_local, var_stage, var_idx, var_config)
    #
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, var_args.workers)) as var_pool:
        for var_config, var_result in var_pool.map(var_worker, list(enumerate(var_configs))):
            var_out.append((var_config, var_result))
    return var_out


def rank_trials(var_pairs, var_metric, var_tie_break):
    """
    [description]
    : rank (config, result) pairs by the metric (max) then the tie-break (min); failed trials last.
    """
    def var_key(var_pair):
        var_result = var_pair[1] or {}
        var_primary = var_result.get(var_metric)
        var_secondary = var_result.get(var_tie_break)
        return (var_primary is not None,
                var_primary if var_primary is not None else float("-inf"),
                -(var_secondary if var_secondary is not None else float("inf")))
    return sorted(var_pairs, key=var_key, reverse=True)


def print_table(var_pairs, var_metric, var_tie_break, var_limit=10):
    """
    [description]
    : print a ranked table of the trials.
    """
    var_keys = sorted(SEARCH_SPACE)
    print("\t".join([var_metric, var_tie_break, "avg_accuracy"] + var_keys))
    for var_config, var_result in var_pairs[:var_limit]:
        var_result = var_result or {}
        var_cells = [f"{var_result.get(var_metric, float('nan')):.4f}",
                     f"{var_result.get(var_tie_break, float('nan')):.4f}",
                     f"{var_result.get('avg_accuracy', float('nan')):.4f}"]
        var_cells += [f"{var_config[var_key]:.5g}" if isinstance(var_config[var_key], float)
                      else str(var_config[var_key]) for var_key in var_keys]
        print("\t".join(var_cells))


#
##
def main():
    """
    [description]
    : run the successive-halving sweep.
    """
    var_args = parse_args()
    os.makedirs(var_args.out, exist_ok=True)
    var_rng = np.random.RandomState(var_args.seed)
    #
    ## stage 1: random search at a short budget
    var_configs = [sample_config(var_rng) for _ in range(var_args.stage1_trials)]
    var_pairs = run_stage(var_args, 1, var_configs, var_args.stage1_epochs, 1)
    if var_args.dry_run:
        return
    var_ranked = rank_trials(var_pairs, var_args.metric, var_args.tie_break)
    print(f"\n--- stage 1 (ranked by {var_args.metric}) ---")
    print_table(var_ranked, var_args.metric, var_args.tie_break)
    #
    ## stage 2: promote the survivors to full length with repeats
    var_configs = [var_config for var_config, _ in var_ranked[:var_args.stage2_trials]]
    var_pairs = run_stage(var_args, 2, var_configs, var_args.stage2_epochs, var_args.stage2_repeats)
    var_ranked = rank_trials(var_pairs, var_args.metric, var_args.tie_break)
    print(f"\n--- stage 2 (ranked by {var_args.metric}) ---")
    print_table(var_ranked, var_args.metric, var_args.tie_break)
    #
    ## stage 3: confirm the top configs with the full repeat count
    var_configs = [var_config for var_config, _ in var_ranked[:var_args.stage3_trials]]
    var_pairs = run_stage(var_args, 3, var_configs, var_args.stage3_epochs, var_args.stage3_repeats)
    var_ranked = rank_trials(var_pairs, var_args.metric, var_args.tie_break)
    print(f"\n--- stage 3 (ranked by {var_args.metric}) ---")
    print_table(var_ranked, var_args.metric, var_args.tie_break)
    #
    var_summary = os.path.join(var_args.out, "summary.json")
    with open(var_summary, "w") as var_file:
        json.dump({"env": var_args.env, "metric": var_args.metric, "tie_break": var_args.tie_break,
                   "best_config": var_ranked[0][0] if var_ranked else None,
                   "best_result": var_ranked[0][1] if var_ranked else None},
                  var_file, indent=4, default=str)
    print(f"\nBest config written to {var_summary}")


if __name__ == "__main__":
    main()
