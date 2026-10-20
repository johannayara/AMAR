#!/usr/bin/env bash
#SBATCH --job-name=diagnose_density
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --qos normal
#SBATCH --time 1:00:00
#SBATCH --gres gpu:a100:1

#
## SLURM wrapper for scripts/diagnose_density_input_dependence.py.
##
## Scores a trained density_map checkpoint on a held-out room under three input conditions
## (real / shuffled / zeroed) and reports how much the predicted map changes. A near-zero relative
## change means the decoder emits the room's marginal map and ignores the CSI input (the dense-BCE
## shortcut); in that regime no input-side preprocessing (per_room_zscore, static_removal, ...) can
## move the cross-domain numbers, so fix input dependence first.
##
## Default fold: train meeting_room+classroom, score empty_room, newest checkpoint under
## visualizations/cross_domain/meeting_room_classroom/. Override any parameter through the environment:
##
##     TRAIN_ENVS=empty_room,classroom TEST_ENV=meeting_room sbatch scripts/diagnose_density_input_dependence.sh
##     CHECKPOINT=visualizations/cross_domain/meeting_room_classroom/7/model.pth \
##         TEST_ENV=empty_room sbatch scripts/diagnose_density_input_dependence.sh
##
## All three leave-one-room-out folds in one submit (per-fold index picks the held-out room and the
## other two become the training rooms); folds whose density checkpoint is missing are reported and
## skipped:
##
##     sbatch --array=0-2 scripts/diagnose_density_input_dependence.sh
##
## Compare preprocessing on/off for a checkpoint by setting PREPROCESS to the same config the
## checkpoint was trained with, e.g. PREPROCESS=zscore or PREPROCESS=static,temporal,zscore.
##
## Environment variables:
##     TRAIN_ENVS   comma-separated training rooms used to locate the checkpoint
##                  (default meeting_room,classroom; ignored when SLURM_ARRAY_TASK_ID is set)
##     TEST_ENV     held-out room whose frames are scored (default empty_room)
##     CHECKPOINT   explicit path to a model.pth; overrides TRAIN_ENVS-based discovery
##     PREPROCESS   value exported as AMAR_CSI_PREPROCESS (default none = raw amplitude)
##     USERS        comma-separated user ids (default 0,1,2,3,4,5)
##     MAX_SAMPLES  frames scored per room, evenly spaced; "all" or empty scores the whole room
##                  (default 600)
##     BATCH_SIZE   forward batch size (default 32)
#
set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

envs=(empty_room meeting_room classroom)

## An array submit picks the held-out room from the index and trains on the other two. Without an
## array the TRAIN_ENVS/TEST_ENV defaults (or overrides) are used.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    TEST_ENV="${envs[$SLURM_ARRAY_TASK_ID]}"
    TRAIN_ENVS=""
    for var_env in "${envs[@]}"; do
        [[ "${var_env}" == "${TEST_ENV}" ]] && continue
        TRAIN_ENVS="${TRAIN_ENVS:+${TRAIN_ENVS},}${var_env}"
    done
fi

TRAIN_ENVS="${TRAIN_ENVS:-meeting_room,classroom}"
TEST_ENV="${TEST_ENV:-empty_room}"
CHECKPOINT="${CHECKPOINT:-}"
PREPROCESS="${PREPROCESS:-none}"
USERS="${USERS:-0,1,2,3,4,5}"
MAX_SAMPLES="${MAX_SAMPLES:-600}"
BATCH_SIZE="${BATCH_SIZE:-32}"

## Resolve the newest checkpoint under the training-room directory. run_cross_domain names the run
## directory after the training rooms joined by "_" and claims an increasing numeric leaf, so the
## highest index is the most recent run.
if [[ -z "${CHECKPOINT}" ]]; then
    var_slug="${TRAIN_ENVS//,/_}"
    var_root="visualizations/cross_domain/${var_slug}"
    CHECKPOINT="$(ls -1d "${var_root}"/*/model.pth 2>/dev/null | sort -V | tail -1 || true)"
    if [[ -z "${CHECKPOINT}" ]]; then
        echo "No density checkpoint under ${var_root} for TRAIN_ENVS=${TRAIN_ENVS}." >&2
        echo "Train one first (scripts/cross_domain_density_map.sh) or pass CHECKPOINT=... explicitly." >&2
        exit 1
    fi
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "CHECKPOINT ${CHECKPOINT} does not exist." >&2
    exit 1
fi

## --max-samples is omitted for "all"/empty so the diagnostic scores the whole room.
max_args=()
if [[ -n "${MAX_SAMPLES}" && "${MAX_SAMPLES}" != "all" ]]; then
    max_args=(--max-samples "${MAX_SAMPLES}")
fi

out_dir="output/diagnose/${TEST_ENV}"
fig_dir="visualizations/diagnose/${TEST_ENV}"
mkdir -p "${out_dir}" "${fig_dir}"
stamp="$(date +%Y%m%d_%H%M%S)"
var_run="$(basename "$(dirname "${CHECKPOINT}")")"
log="${out_dir}/diagnose_${TEST_ENV}_${var_slug:-manual}_${var_run}_${stamp}.log"

export WANDB_MODE=offline
export AMAR_CSI_PREPROCESS="${PREPROCESS}"
export PYTHONUNBUFFERED=1

echo "Diagnostic | checkpoint=${CHECKPOINT} | test_env=${TEST_ENV} | preprocess=${PREPROCESS}"
echo "Log: ${log}"

start=$(date +%s)
python scripts/diagnose_density_input_dependence.py \
    --checkpoint "${CHECKPOINT}" \
    --env "${TEST_ENV}" \
    --users "${USERS}" \
    --batch-size "${BATCH_SIZE}" \
    "${max_args[@]}" \
    --out "${fig_dir}/diagnose_${TEST_ENV}_${var_run}_${stamp}.png" \
    2>&1 | tee "${log}"
end=$(date +%s)
echo "Total runtime: $((end - start)) seconds"
