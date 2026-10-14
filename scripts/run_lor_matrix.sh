#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 16G
#SBATCH --qos normal
#SBATCH --time 12:00:00
#SBATCH --gres gpu:a100:1
#SBATCH --array=0-11

## Leave-one-room-out (LoR) matrix over {AMAR_WO_RVQ, density_map} x backbone x channel norm.
##
## Same axes as scripts/run_encoder_matrix.sh, but the LoR scenario instead of the single-room
## `run_main.py` one: each combo trains on two WiMANS rooms and tests on the third (the held-out
## room contributes no labels), via scripts/run_cross_domain.py.
##
##   * model      : AMAR_WO_RVQ | density_map   (the real AMAR_WO_RVQ set-prediction model)
##   * backbone   : nn.backbone           default | thatv-nostats   <- the two relevant ones
##   * encoder    : nn.encoder_norm       batch                     <- see note below
##   * fold       : held-out room         empty_room | meeting_room | classroom
##
## Which combos are "relevant" is taken from output/matrix (the single-room matrix runs):
##   * AMAR_WO_RVQ: default (err 0.610) and thatv-nostats (err 1.145) dominate; thatv (3.167) and
##     thatv-nopool (5.591) collapse, and thatv-notokens cannot back AMAR_WO_RVQ at all
##     (forward_sequence needs the length-preserving convs). Kept: default, thatv-nostats.
##   * density_map (density head): default (countMAE 0.478) and thatv-nostats (0.455) are the two
##     non-dominated backbones. Kept: default, thatv-nostats.
##   * encoder: the single-room matrix's `batch|group|layer` axis was a no-op — it passed
##     `--hp nn.encoder=...`, but the code reads `nn.encoder_norm` / $AMAR_ENCODER_NORM
##     (src/models/bce_that.py), so every triple was bit-identical. This script passes the real key
##     (`--hp nn.encoder_norm=...`), defaulting to the preset value `batch`; set ENCODERS="batch group"
##     to turn it into a live room-agnostic-norm axis.
##
## Two ways to run, both parallel:
##   * SLURM array (one combo per task, spreads across nodes): the `--array=0-11` above equals the
##     combo count minus one (see `--list`). Submit with `sbatch scripts/run_lor_matrix.sh`, or
##     override the range: `sbatch --array=0-3 scripts/run_lor_matrix.sh`.
##   * locally (bounded background parallelism): `MAX_JOBS=4 ./scripts/run_lor_matrix.sh`.
##
## Each combo is isolated in output/lor_matrix/<model>/<backbone>/<encoder>/holdout_<room>/ and
## visualizations/cross_domain/<train_rooms>/<n>/ (run_cross_domain claims the numeric leaf
## atomically, so concurrent runs cannot collide).
##
## Examples:
##   ./scripts/run_lor_matrix.sh --list                     # print combos + array range, run nothing
##   EPOCHS=50 REPEAT=3 MAX_JOBS=4 ./scripts/run_lor_matrix.sh
##   ENCODERS="batch group" MAX_JOBS=6 ./scripts/run_lor_matrix.sh
##   sbatch scripts/run_lor_matrix.sh
#
set -euo pipefail

REPEAT="${REPEAT:-3}"
EPOCHS="${EPOCHS:-30}"
USERS="${USERS:-0,1,2,3,4,5}"
MAX_JOBS="${MAX_JOBS:-4}"

models=(${MODELS:-AMAR_WO_RVQ density_map})
backbones=(${BACKBONES:-default thatv-nostats})
encoders=(${ENCODERS:-batch})
all_envs=(${ENVS:-empty_room meeting_room classroom})

export WANDB_MODE=offline

## ---- build the combo list (model|backbone|encoder|holdout|train_envs) ---------------------------
combos=()
add_combo() { combos+=("$1|$2|$3|$4|$5"); }

for model in "${models[@]}"; do
    for backbone in "${backbones[@]}"; do
        ## AMAR_WO_RVQ's default CNN Backbone ignores the channel norm, so one encoder is enough there
        for encoder in "${encoders[@]}"; do
            for holdout in "${all_envs[@]}"; do
                train_envs=""
                for env in "${all_envs[@]}"; do
                    if [[ "${env}" != "${holdout}" ]]; then
                        train_envs="${train_envs:+${train_envs},}${env}"
                    fi
                done
                add_combo "${model}" "${backbone}" "${encoder}" "${holdout}" "${train_envs}"
            done
        done
    done
done

if [[ "${1:-}" == "--list" ]]; then
    var_idx=0
    for combo in "${combos[@]}"; do echo "${var_idx}: ${combo}"; var_idx=$((var_idx + 1)); done
    echo "${#combos[@]} combos -> sbatch --array=0-$(( ${#combos[@]} - 1 ))"
    exit 0
fi

## the runs need the AMAR env;
if [[ -f /software/anaconda3/etc/profile.d/conda.sh ]]; then
    source /software/anaconda3/etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate AMAR

## ---- one combo -----------------------------------------------------------------------------------
run_combo() {
    local combo="$1"
    local model backbone encoder holdout train_envs
    IFS='|' read -r model backbone encoder holdout train_envs <<< "${combo}"
    local log_dir="./output/lor_matrix/${model}/${backbone}/${encoder}/holdout_${holdout}"
    mkdir -p "${log_dir}"
    echo "=== model=${model} backbone=${backbone} encoder_norm=${encoder} holdout=${holdout} train=${train_envs}"
    python scripts/run_cross_domain.py \
        --model "${model}" --task location --repeat "${REPEAT}" --users "${USERS}" \
        --train_envs "${train_envs}" --epochs "${EPOCHS}" \
        --hp "nn.backbone=${backbone}" --hp "nn.encoder_norm=${encoder}" \
        > "${log_dir}/run_$(date +%Y%m%d_%H%M%S).log" 2>&1
}

## ---- SLURM array: this task runs exactly one combo -----------------------------------------------
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    if (( SLURM_ARRAY_TASK_ID >= ${#combos[@]} )); then
        echo "array index ${SLURM_ARRAY_TASK_ID} >= combo count ${#combos[@]}; nothing to do"
        exit 0
    fi
    run_combo "${combos[$SLURM_ARRAY_TASK_ID]}"
    exit 0
fi

## ---- local: bounded background parallelism -------------------------------------------------------
echo "Running ${#combos[@]} combos, up to ${MAX_JOBS} at a time"
var_fail=0
for combo in "${combos[@]}"; do
    run_combo "${combo}" &
    while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
        if ! wait -n; then var_fail=$((var_fail + 1)); fi
    done
done
while (( $(jobs -rp | wc -l) > 0 )); do
    if ! wait -n; then var_fail=$((var_fail + 1)); fi
done

echo "LoR matrix done: ${#combos[@]} combos, ${var_fail} failed (see each output/lor_matrix/<...>/run_*.log)"
