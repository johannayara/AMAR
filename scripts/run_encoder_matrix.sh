#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 16G
#SBATCH --qos normal
#SBATCH --time 12:00:00
#SBATCH --gres gpu:a100:1
#SBATCH --array=0-39

## Matrix ablation over {AMAR_WO_RVQ, density_map} x head x encoder backbone x channel norm.
##
##   * model      : AMAR_WO_RVQ | density_map
##   * head       : density | coords      (density_map only; AMAR_WO_RVQ has no head switch)
##   * backbone   : nn.backbone           default | thatv | thatv-nostats | thatv-nopool | thatv-notokens
##   * encoder    : AMAR_ENCODER_NORM     batch | group | layer   (channel norm inside the THAT/THATV
##                                         Encoder; it is a no-op for AMAR_WO_RVQ's default CNN
##                                         Backbone, which does not read it)
##
## Two ways to run, both parallel:
##   * SLURM array (one combo per task, spreads across nodes): the `--array=0-39` above equals the
##     combo count minus one (see `--list`). Submit with `sbatch scripts/run_encoder_matrix.sh`, or
##     override the range: `sbatch --array=0-9 scripts/run_encoder_matrix.sh`.
##   * locally (bounded background parallelism): `MAX_JOBS=4 ./scripts/run_encoder_matrix.sh`.
##
## Each combo is isolated in output/<model>/<head>/<backbone>/<encoder>/<env>/ and
## visualizations/<model>/<backbone>/<env>/<n>/ (run_main claims the numeric leaf atomically, so
## concurrent runs cannot collide).
##
## Examples:
##   ./scripts/run_encoder_matrix.sh --list                 # print combos + array range, run nothing
##   ENV=meeting_room EPOCHS=50 REPEAT=3 MAX_JOBS=4 ./scripts/run_encoder_matrix.sh
##   sbatch scripts/run_encoder_matrix.sh
#
set -euo pipefail

ENV=classroom
REPEAT="${REPEAT:-3}"
EPOCHS="${EPOCHS:-30}"
USERS="${USERS:-0,1,2,3,4,5}"
MAX_JOBS="${MAX_JOBS:-4}"

models=(AMAR_WO_RVQ density_map)
heads=(density coords)
backbones=(default thatv thatv-nostats thatv-nopool thatv-notokens)
encoders=(batch group layer)

export WANDB_MODE=offline

## ---- build the combo list (model|head|backbone|encoder) -----------------------------------------
combos=()
add_combo() { combos+=("$1|$2|$3|$4"); }

for model in "${models[@]}"; do
    for backbone in "${backbones[@]}"; do
        if [[ "${model}" == "AMAR_WO_RVQ" ]]; then
            ## forward_sequence needs the length-preserving convs; thatv-notokens only supports forward()
            if [[ "${backbone}" == "thatv-notokens" ]]; then
                echo "skip AMAR_WO_RVQ backbone=${backbone}: forward_sequence requires the tokens feature" >&2
                continue
            fi
            if [[ "${backbone}" == "default" ]]; then
                ## the CNN Backbone ignores encoder_norm, so one run is enough
                add_combo "${model}" "amar" "${backbone}" "batch"
            else
                for encoder in "${encoders[@]}"; do add_combo "${model}" "amar" "${backbone}" "${encoder}"; done
            fi
        else
            for head in "${heads[@]}"; do
                for encoder in "${encoders[@]}"; do add_combo "${model}" "${head}" "${backbone}" "${encoder}"; done
            done
        fi
    done
done

if [[ "${1:-}" == "--list" ]]; then
    var_idx=0
    for combo in "${combos[@]}"; do echo "${var_idx}: ${combo}"; var_idx=$((var_idx + 1)); done
    echo "${#combos[@]} combos -> sbatch --array=0-$(( ${#combos[@]} - 1 ))"
    exit 0
fi

## the runs need the AMAR env; 
source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR

## ---- one combo -----------------------------------------------------------------------------------
run_combo() {
    local combo="$1"
    local model head backbone encoder
    IFS='|' read -r model head backbone encoder <<< "${combo}"
    local log_dir="./output/matrix/${model}/${head}/${backbone}/${encoder}/${ENV}"
    mkdir -p "${log_dir}"
    local extra=()
    if [[ "${model}" == "density_map" ]]; then extra=(--hp "density.head=${head}"); fi
    echo "=== model=${model} head=${head} backbone=${backbone} encoder_norm=${encoder} env=${ENV}"
    python scripts/run_main.py \
        --model "${model}" --task location --repeat "${REPEAT}" --env "${ENV}"\
        --epochs "${EPOCHS}" \
        --hp "nn.backbone=${backbone}" "${extra[@]}" --hp "nn.encoder=${encoder}"\
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

echo "Matrix done: ${#combos[@]} combos, ${var_fail} failed (see each output/<...>/run_*.log)"
