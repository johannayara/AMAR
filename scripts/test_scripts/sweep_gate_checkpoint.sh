#!/usr/bin/env bash
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --qos normal
#SBATCH --time 2:00:00
#SBATCH --gres gpu:a100:1
#SBATCH --array=0-5
##
## Environment variables:
##     COMBOS       newline- or ';'-separated "<checkpoint>|<test room>|<train rooms>" overrides
##     GATES        comma-separated gate values          (default off,calibrated,0.01..0.30)
##     PEAK_RULE    new | old                            (default new; old reproduces the pre-fix
##                                                        position metrics)
##     USERS        comma-separated user IDs             (default "0, 1,2,3,4,5")
##     BACKBONE     override nn.backbone                 (default: the value in the checkpoint)
##     ENCODER_NORM override nn.encoder_norm             (default: the value in the checkpoint)
##     OUT_ROOT     output root                          (default output/gate_sweep)
##     MAX_JOBS     local parallelism                    (default 4)
##
## Each combo writes <OUT_ROOT>/<tag>/run.log and <OUT_ROOT>/<tag>/gate_sweep.json.
#
set -euo pipefail

## GATES is passed through only when set, so the python default (off,calibrated,peakThr,q05..q90,
## anchored on the effective range) is the single source of truth
GATES="${GATES:-}"
PEAK_RULE="${PEAK_RULE:-new}"
USERS="${USERS:-0, 1,2,3,4,5}"
MAX_JOBS="${MAX_JOBS:-4}"
OUT_ROOT="${OUT_ROOT:-output/gate_sweep}"

## ---- combo list: checkpoint | test room | training rooms ----------------------------------------
## Defaults are the checkpoints behind the runs discussed so far, all of which exist on disk.
combos=()
if [[ -n "${COMBOS:-}" ]]; then
    while IFS= read -r var_line; do
        var_line="${var_line%%#*}"
        var_line="$(echo "${var_line}" | xargs)"
        [[ -n "${var_line}" ]] && combos+=("${var_line}")
    done < <(printf '%s\n' "${COMBOS}" | tr ';' '\n')
else
    ## the canonical cross-domain baseline: default density_map (THAT backbone, density head), Adam,
    ## BCE loss (run_density_map_cross_domain's default var_loss_mode), layout_free. Same fold and the
    ## same 150 epochs as the sgd and dem runs below, so the three are directly comparable.
    ## 36 and 37 are that config from two dates: 36 ran with the empty gate calibrated (its log carries
    ## finite EmptyThr values) and 37 with the gate off in effect (EmptyThr nan), so the pair shows how
    ## much the selection epoch, not just the sweep's decision, moves these numbers. 21/23/24 are the
    ## 10-26 LoR-matrix default runs at 50 epochs on the current code, for a current-code reference.
    combos+=("visualizations/cross_domain/empty_room_classroom/36/model.pth|meeting_room|empty_room,classroom")
    combos+=("visualizations/cross_domain/empty_room_classroom/37/model.pth|meeting_room|empty_room,classroom")
    combos+=("visualizations/cross_domain/empty_room_classroom/33/model.pth|meeting_room|empty_room,classroom")
    combos+=("visualizations/cross_domain/empty_room_classroom/44/model.pth|meeting_room|empty_room,classroom")
    combos+=("visualizations/density_map/default/classroom/1/model.pth|classroom|classroom")
    combos+=("visualizations/density_map/thatv-nostats/classroom/4/model.pth|classroom|classroom")
fi

if [[ "${1:-}" == "--list" ]]; then
    var_idx=0
    for combo in "${combos[@]}"; do echo "${var_idx}: ${combo}"; var_idx=$((var_idx + 1)); done
    echo "${#combos[@]} combos -> sbatch --array=0-$(( ${#combos[@]} - 1 ))"
    exit 0
fi

## the runs need the AMAR env
if [[ -f /software/anaconda3/etc/profile.d/conda.sh ]]; then
    source /software/anaconda3/etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate AMAR

## ---- one combo -----------------------------------------------------------------------------------
run_combo() {
    local combo="$1"
    local checkpoint env train_envs
    IFS='|' read -r checkpoint env train_envs <<< "${combo}"
    local tag
    tag="$(echo "${checkpoint%.pth}" | sed 's|^visualizations/||; s|/model$||' | tr '/.' '__')_${env}"
    local out_dir="${OUT_ROOT}/${tag}"
    mkdir -p "${out_dir}"
    if [[ ! -f "${checkpoint}" ]]; then
        echo "=== skip ${combo}: no such checkpoint ${checkpoint}" | tee "${out_dir}/run.log"
        return 0
    fi
    local extra=()
    [[ -n "${BACKBONE:-}" ]] && extra+=(--backbone "${BACKBONE}")
    [[ -n "${ENCODER_NORM:-}" ]] && extra+=(--encoder-norm "${ENCODER_NORM}")
    [[ -n "${GATES}" ]] && extra+=(--gates "${GATES}")
    echo "=== checkpoint=${checkpoint} env=${env} train=${train_envs} gates=${GATES:-<default>} peak_rule=${PEAK_RULE}"
    python scripts/sweep_gate_checkpoint.py \
        --checkpoint "${checkpoint}" --env "${env}" --train_envs "${train_envs}" \
        --users "${USERS}" --peak_rule "${PEAK_RULE}" \
        --out "${out_dir}/gate_sweep.json" "${extra[@]}" \
        > "${out_dir}/run.log" 2>&1
    echo "  -> ${out_dir}/run.log"
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

echo "Gate sweep done: ${#combos[@]} combos, ${var_fail} failed (see ${OUT_ROOT}/<tag>/run.log)"
