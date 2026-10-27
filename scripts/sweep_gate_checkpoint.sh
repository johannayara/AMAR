#!/usr/bin/env bash
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --qos normal
#SBATCH --time 2:00:00
#SBATCH --gres gpu:a100:1
#SBATCH --array=0-3

#
## Does the stage-1 empty gate do anything? Offline, on saved checkpoints, no retraining.
##
## One array task per checkpoint: reload it, rebuild the run's validation split, run inference on the
## held-out room once, then score the SAME predicted maps under a list of gate values
## (scripts/sweep_gate_checkpoint.py). The "off" row must reproduce the run's reported numbers up to the
## last-repeat-vs-aggregate difference; the other rows price the gate.
##
## Every run committed up to 2026-10-27 had the gate disabled (preset density.staged_count was None,
## read as falsy), which shows in those logs as "EmptyThr nan" and in their per-count tables as a near
## zero accuracy on the true-empty class. This is the measurement that says whether enabling it is
## worth a re-run.
##
## Combo count = ${#combos[@]}, so --array must be 0-(count-1). After editing the combo list below or
## COMBOS, run `scripts/sweep_gate_checkpoint.sh --list` and set --array to the range it prints -- the
## header cannot be computed.
##
## Submit from the repository root:
##     sbatch scripts/sweep_gate_checkpoint.sh
##     sbatch --array=0-1 scripts/sweep_gate_checkpoint.sh
##     MAX_JOBS=2 ./scripts/sweep_gate_checkpoint.sh          # locally, bounded parallelism
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

GATES="${GATES:-off,calibrated,0.01,0.02,0.04,0.06,0.08,0.10,0.15,0.20,0.30}"
PEAK_RULE="${PEAK_RULE:-new}"
USERS="${USERS:-0, 1,2,3,4,5}"
MAX_JOBS="${MAX_JOBS:-4}"
OUT_ROOT="${OUT_ROOT:-output/gate_sweep}"

## ---- combo list: checkpoint | test room | training rooms ----------------------------------------
## Defaults are the four checkpoints behind the runs discussed so far, all of which exist on disk.
combos=()
if [[ -n "${COMBOS:-}" ]]; then
    while IFS= read -r var_line; do
        var_line="${var_line%%#*}"
        var_line="$(echo "${var_line}" | xargs)"
        [[ -n "${var_line}" ]] && combos+=("${var_line}")
    done < <(printf '%s\n' "${COMBOS}" | tr ';' '\n')
else
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
    echo "=== checkpoint=${checkpoint} env=${env} train=${train_envs} gates=${GATES} peak_rule=${PEAK_RULE}"
    python scripts/sweep_gate_checkpoint.py \
        --checkpoint "${checkpoint}" --env "${env}" --train_envs "${train_envs}" \
        --users "${USERS}" --gates "${GATES}" --peak_rule "${PEAK_RULE}" \
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
