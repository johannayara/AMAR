#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 16G
#SBATCH --qos normal
#SBATCH --time 12:00:00
#SBATCH --gres gpu:a100:1

## Matrix ablation over {AMAR_WO_RVQ, density_map} x head x encoder backbone x channel norm.
##
##   * model      : AMAR_WO_RVQ | density_map
##   * head       : density | coords      (density_map only; AMAR_WO_RVQ has no head switch)
##   * backbone   : nn.backbone           default | thatv | thatv-nostats | thatv-nopool | thatv-notokens
##   * encoder    : AMAR_ENCODER_NORM     batch | group | layer   (channel norm inside the THAT/THATV
##                                         Encoder; it is a no-op for AMAR_WO_RVQ's default CNN
##                                         Backbone, which does not read it)
##
## Each run is isolated in output/<model>/<head>/<backbone>/<encoder>/<env>/, mirroring the
## visualizations/<model>/<backbone>/<env>/<n> tree run_main writes. Run it directly, or submit it
## with sbatch (the loops are sequential; parallelise by splitting the arrays and submitting several
## copies with SLICE/NSLICES below).
##
## Examples:
##   ./scripts/run_encoder_matrix.sh
##   ENV=meeting_room EPOCHS=50 REPEAT=3 ./scripts/run_encoder_matrix.sh
##   SLICE=0 NSLICES=4 ./scripts/run_encoder_matrix.sh      # every 4th combo, for 4 parallel jobs
#
set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR

ENV="${ENV:-classroom}"
REPEAT="${REPEAT:-3}"
EPOCHS="${EPOCHS:-50}"
USERS="${USERS:-0,1,2,3,4,5}"
SLICE="${SLICE:-0}"
NSLICES="${NSLICES:-1}"

models=(AMAR_WO_RVQ density_map)
heads=(density coords)
backbones=(default thatv thatv-nostats thatv-nopool thatv-notokens)
encoders=(batch group layer)

export WANDB_MODE=offline

var_index=0

run_one() {
    local model="$1" head="$2" backbone="$3" encoder="$4"
    ## round-robin slicing so several sbatch copies split the matrix without coordination
    if (( var_index % NSLICES != SLICE )); then var_index=$((var_index + 1)); return 0; fi
    var_index=$((var_index + 1))
    local log_dir="./output/${model}/${head}/${backbone}/${encoder}/${ENV}"
    mkdir -p "${log_dir}"
    local extra=()
    if [[ "${model}" == "density_map" ]]; then extra=(--hp "density.head=${head}"); fi
    echo "=== model=${model} head=${head} backbone=${backbone} encoder_norm=${encoder} env=${ENV}"
    AMAR_ENCODER_NORM="${encoder}" python scripts/run_main.py \
        --model "${model}" --task location --repeat "${REPEAT}" --env "${ENV}" --users "${USERS}" \
        --epochs "${EPOCHS}" \
        --hp "nn.backbone=${backbone}" "${extra[@]}" \
        > "${log_dir}/run_$(date +%Y%m%d_%H%M%S).log" 2>&1
}

for model in "${models[@]}"; do
    for backbone in "${backbones[@]}"; do
        if [[ "${model}" == "AMAR_WO_RVQ" ]]; then
            ## forward_sequence needs the length-preserving convs; thatv-notokens only supports forward()
            if [[ "${backbone}" == "thatv-notokens" ]]; then
                echo "skip AMAR_WO_RVQ backbone=${backbone}: forward_sequence requires the tokens feature"
                continue
            fi
            if [[ "${backbone}" == "default" ]]; then
                ## the CNN Backbone ignores encoder_norm, so one run is enough
                run_one "${model}" "amar" "${backbone}" "batch"
            else
                for encoder in "${encoders[@]}"; do run_one "${model}" "amar" "${backbone}" "${encoder}"; done
            fi
        else
            for head in "${heads[@]}"; do
                for encoder in "${encoders[@]}"; do run_one "${model}" "${head}" "${backbone}" "${encoder}"; done
            done
        fi
    done
done

echo "Matrix done ($((var_index)) combos considered)."
