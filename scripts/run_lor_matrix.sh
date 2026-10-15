#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 32G
#SBATCH --qos normal
#SBATCH --time 12:00:00
#SBATCH --gres gpu:a100:1
#SBATCH --array=0-35

## Leave-one-room-out (LoR) matrix over model x head x backbone x channel norm.
##
## Same axes as scripts/run_encoder_matrix.sh, but the LoR scenario instead of the single-room
## `run_main.py` one: each combo trains on two WiMANS rooms and tests on the third (the held-out
## room contributes no labels), via scripts/run_cross_domain.py.
##
##   * model      : MODELS               default: density_map
##   * head       : density.head         coords | density            (density_map only)
##   * backbone   : nn.backbone          default | thatv-nostats
##   * encoder    : nn.encoder_norm      batch | group | layer
##   * fold       : held-out room        empty_room | meeting_room | classroom
##
## Combo count = |MODELS| x |HEADS| x |BACKBONES| x |ENCODERS| x |ENVS|
##             = 1 x 2 x 2 x 3 x 3 = 36, so --array must be 0-35.
## After changing any axis, run `scripts/run_lor_matrix.sh --list` and set --array to the range it
## prints -- the header cannot be computed.
##
## Two caveats about the axes, from reading the model code:
##   * nn.encoder_norm is honoured only by the THAT/THATV encoders (bce_that.make_channel_norm). With
##     nn.backbone=cnn the AMAR CNN Backbone hardcodes BatchNorm1d, so batch/group/layer are identical.
##     Every density_map variant (density head, coords head, dem) accepts default/thatv*/cnn as a
##     backbone; cnn is simply the one backbone for which the encoder axis is a no-op.
##   * density.head is meaningful only for the density_map model; the other models ignore it.
set -euo pipefail

REPEAT="${REPEAT:-3}"
EPOCHS="${EPOCHS:-50}"
USERS="${USERS:-0,1,2,3,4,5}"
MAX_JOBS="${MAX_JOBS:-4}"

models=(${MODELS:-density_map})
HEADS=(${HEADS_OVERRIDE:-coords density})
backbones=(${BACKBONES:-default thatv-nostats})
encoders=(${ENCODERS:-batch group layer})
all_envs=(${ENVS:-empty_room meeting_room classroom})

export WANDB_MODE=offline

## ---- build the combo list (model|backbone|encoder|holdout|train_envs|head) ----------------------
combos=()
add_combo() { combos+=("$1|$2|$3|$4|$5|$6"); }

for model in "${models[@]}"; do
    for backbone in "${backbones[@]}"; do
        for head in "${HEADS[@]}"; do
            for encoder in "${encoders[@]}"; do
                for holdout in "${all_envs[@]}"; do
                    train_envs=""
                    for env in "${all_envs[@]}"; do
                        if [[ "${env}" != "${holdout}" ]]; then
                            train_envs="${train_envs:+${train_envs},}${env}"
                        fi
                    done
                    add_combo "${model}" "${backbone}" "${encoder}" "${holdout}" "${train_envs}" "${head}"
                done
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
    local model backbone encoder holdout train_envs head
    IFS='|' read -r model backbone encoder holdout train_envs head <<< "${combo}"
    ## the head is part of the path: coords and density would otherwise share a log directory
    local log_dir="./output/lor_matrix/${model}/${head}/${backbone}/${encoder}/holdout_${holdout}"
    mkdir -p "${log_dir}"
    echo "=== model=${model} head=${head} backbone=${backbone} encoder_norm=${encoder} holdout=${holdout} train=${train_envs}"
    python scripts/run_cross_domain.py \
        --model "${model}" --task location --repeat "${REPEAT}" --users "${USERS}" \
        --train_envs "${train_envs}" --epochs "${EPOCHS}" \
        --hp "nn.backbone=${backbone}" --hp "nn.encoder_norm=${encoder}" \
        --hp "density.head=${head}" \
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
