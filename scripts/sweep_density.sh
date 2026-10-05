#!/usr/bin/env bash
#SBATCH --job-name=density_sweep
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --qos normal
#SBATCH --time 12:00:00
#SBATCH --gres gpu:a100:1
#SBATCH --array=0-2
#SBATCH --output=slurm_density_sweep_%A_%a.out
#
## Successive-halving hyperparameter sweep of the density-map model, one room per array task.
## Thin wrapper around scripts/sweep_density.py (left unmodified): every trial is its own
## `python scripts/run_main.py --model density_map ...` process, so a crashed trial cannot poison
## the others.
##
## Submit from the repository root:
##     sbatch scripts/sweep_density.sh                     # all three rooms, 1 GPU each
##     sbatch --array=0 scripts/sweep_density.sh           # empty_room only
##     sbatch --array=1-2 scripts/sweep_density.sh         # meeting_room + classroom only
##
## Extra flags after the script name are forwarded to sweep_density.py, e.g.:
##     sbatch scripts/sweep_density.sh --calibrated_threshold --resume
##     sbatch scripts/sweep_density.sh --stage1_trials 12 --stage2_trials 4
##
## Environment variables (set with `sbatch --export=ALL,METRIC=... scripts/sweep_density.sh`,
## or `METRIC=... sbatch scripts/sweep_density.sh`):
##     METRIC      ranking metric, max        (default avg_balanced_accuracy)
##     TIE_BREAK   tie-break metric, min      (default avg_mae)
##     SEED        random-search seed         (default 39)
##     WORKERS     parallel trials per task   (default 1)
##     GPUS        GPU ids for the workers    (default 0)
##
## To sweep ONE room across several GPUs instead of one room per task: submit with --array=0, raise
## the GPU request (edit --gres above to gpu:a100:4, or pass --gres=gpu:a100:4 on the sbatch line),
## then run with WORKERS=4 GPUS=0,1,2,3.
##
## Preview the trial commands without running anything (no GPU needed):
##     python scripts/sweep_density.py --env empty_room \
##         --out output/sweep/density_map_empty_room --dry_run
##
## Current limitations of scripts/sweep_density.py — NOT worked around here, since it must stay
## unmodified:
##   * the model is hardcoded to `density_map`; sweeping the DEM variant needs that literal changed
##     inside run_trial().
##   * `density.eval_threshold` is pinned to 0.5 unless --calibrated_threshold is passed, but the new
##     `density.staged_count` default (True) still calibrates the empty gate per trial on the split
##     that is then scored, so the empty class can be over-fitted during the sweep. Neither
##     `staged_count` nor `empty_threshold` is reachable from the command line; add them to
##     SEARCH_SPACE (or pin them) inside sweep_density.py if you want them controlled.
##   * trials are isolated only in their JSON/out dir; the figure+checkpoint dirs written under
##     visualizations/density_map/<env>/<n>/ are not redirected, so a full sweep leaves roughly
##     20 MB per trial there. Clean that tree up between sweeps if disk matters.
#
set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

envs=(empty_room meeting_room classroom)
env=${envs[${SLURM_ARRAY_TASK_ID:-0}]}
out="output/sweep/density_map_${env}"

mkdir -p "$out"

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

start=$(date +%s)
echo "Start ${env} on $(hostname): $(date)"
echo "Out dir: ${out} | workers ${WORKERS:-1} | gpus ${GPUS:-0}"

python scripts/sweep_density.py \
    --env "$env" \
    --out "$out" \
    --metric "${METRIC:-avg_balanced_accuracy}" \
    --tie_break "${TIE_BREAK:-avg_mae}" \
    --seed "${SEED:-39}" \
    --workers "${WORKERS:-1}" \
    --gpus "${GPUS:-0}" \
    "$@" \
    > "${out}/sweep.log" 2>&1

echo "Total runtime ${env}: $(( $(date +%s) - start )) seconds"
