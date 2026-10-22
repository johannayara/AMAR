#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 32G
#SBATCH --qos normal
#SBATCH --time 12:00:00
#SBATCH --gres gpu:a100:1
#SBATCH --array=0-15

## Few-shot H-WILD -> WiMANS hyperparameter sweep (scripts/run_few_shot_hwild_to_wimans.py).
##
## Axes (all overridable through the environment):
##   * KD_WEIGHTS        few_shot.kd_weight    default "0 0.1 0.3 1.0"
##   * FEW_SHOT_RATIOS   few_shot.few_shot_ratio   default "0.05 0.10"
##   * INIT_MODES        teacher | random      default "teacher random"
##   * STANDARDIZE_MODES window | none         default "window"
##
## Combo count = |KD_WEIGHTS| x |FEW_SHOT_RATIOS| x |INIT_MODES| x |STANDARDIZE_MODES|
##             = 4 x 2 x 2 x 1 = 16, so --array must be 0-15.
## After changing any axis, run `scripts/sweep_few_shot_hwild.sh --list` and set --array to the
## range it prints -- the header cannot be computed.
##
## Why these axes (from the existing runs):
##   * kd_weight is the one that moved the needle: kd=1 pulls the student toward the H-WILD teacher,
##     whose WiMANS error is worse than the constant baseline, so kd=0 won on every room. The grid is
##     therefore centred on small kd rather than on kd=1.
##   * init_from_teacher is the second "distillation" channel (warm start), so it is swept against kd:
##     kd=0 + init=teacher is still a form of transfer, kd=0 + init=random is the pure few-shot floor.
##   * standardize is the one preprocessing knob that changes the teacher, so it is swept separately.
##
## Cost: the runner retrains the H-WILD teacher on every invocation, and the teacher does NOT depend
## on kd / few_shot_ratio / init, only on (train rooms, window, stride, standardize, teacher epochs).
## So every combo pays a full teacher training. Lower TEACHER_EPOCHS (the teacher early-stops around
## epoch 110-135 of 200 on the full set) or bound the data with MAX_FILES/MAX_SAMPLES to keep a first
## pass cheap, then re-run the winners at full length.
##
## Two ways to run, both parallel:
##   * SLURM array (one combo per task, spreads across nodes): `sbatch scripts/sweep_few_shot_hwild.sh`,
##     or override the range: `sbatch --array=0-3 scripts/sweep_few_shot_hwild.sh`.
##   * locally (bounded background parallelism): `MAX_JOBS=4 ./scripts/sweep_few_shot_hwild.sh`.
##
## Each combo writes its log, JSON and TXT report under output/sweep/few_shot_hwild_wimans/<tag>/, so
## the collector sees only sweep outputs and repeated sweeps never overwrite each other.
##
## Examples:
##   ./scripts/sweep_few_shot_hwild.sh --list
##   KD_WEIGHTS="0 0.1 0.3 1.0" MAX_JOBS=4 ./scripts/sweep_few_shot_hwild.sh
##   MAX_FILES=8 MAX_SAMPLES=400 TEACHER_EPOCHS=60 REPEAT=1 ./scripts/sweep_few_shot_hwild.sh
##   python scripts/sweep_collect_few_shot.py
#
set -euo pipefail

TRAIN_ROOMS="${TRAIN_ROOMS:-Conference,Laboratory,classroom}"
TEST_ROOMS="${TEST_ROOMS:-meeting_room,Lounge,Office,empty_room}"
REPEAT="${REPEAT:-3}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-200}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-200}"
KD_WEIGHTS="${KD_WEIGHTS:-0 0.1 0.3 1.0}"
FEW_SHOT_RATIOS="${FEW_SHOT_RATIOS:-0.05 0.10}"
INIT_MODES="${INIT_MODES:-teacher random}"
STANDARDIZE_MODES="${STANDARDIZE_MODES:-window}"
WINDOW="${WINDOW:-}"
STRIDE="${STRIDE:-}"
MAX_FILES="${MAX_FILES:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
USERS="${USERS:-}"
INTERFERENCE="${INTERFERENCE:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
MAX_JOBS="${MAX_JOBS:-4}"
OUT_ROOT="${OUT_ROOT:-output/sweep/few_shot_hwild_wimans}"

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

read -r -a kd_weights <<< "${KD_WEIGHTS}"
read -r -a few_shot_ratios <<< "${FEW_SHOT_RATIOS}"
read -r -a init_modes <<< "${INIT_MODES}"
read -r -a standardize_modes <<< "${STANDARDIZE_MODES}"

## ---- build the combo list (kd|few_shot_ratio|init|standardize) ----------------------------------
combos=()
for kd in "${kd_weights[@]}"; do
    for fs in "${few_shot_ratios[@]}"; do
        for init in "${init_modes[@]}"; do
            for std in "${standardize_modes[@]}"; do
                combos+=("${kd}|${fs}|${init}|${std}")
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
    local kd fs init std
    IFS='|' read -r kd fs init std <<< "${combo}"
    local tag="kd${kd}_fs${fs}_init${init}_std${std}"
    local log_dir="${OUT_ROOT}/${tag}"
    mkdir -p "${log_dir}"
    local args=(--train_rooms "${TRAIN_ROOMS}" --test_rooms "${TEST_ROOMS}"
                --few_shot_ratio "${fs}" --kd_weight "${kd}" --repeat "${REPEAT}"
                --teacher_epochs "${TEACHER_EPOCHS}" --student_epochs "${STUDENT_EPOCHS}"
                --standardize "${std}")
    if [[ "${init}" == "random" ]]; then args+=(--no_init_from_teacher); fi
    if [[ -n "${WINDOW}" ]]; then args+=(--window "${WINDOW}"); fi
    if [[ -n "${STRIDE}" ]]; then args+=(--stride "${STRIDE}"); fi
    if [[ -n "${MAX_FILES}" ]]; then args+=(--max_files "${MAX_FILES}"); fi
    if [[ -n "${MAX_SAMPLES}" ]]; then args+=(--max_samples "${MAX_SAMPLES}"); fi
    if [[ -n "${USERS}" ]]; then args+=(--users "${USERS}"); fi
    if [[ -n "${INTERFERENCE}" ]]; then args+=(--interference "${INTERFERENCE}"); fi
    if [[ -n "${EXTRA_ARGS}" ]]; then read -r -a var_extra <<< "${EXTRA_ARGS}"; args+=("${var_extra[@]}"); fi
    echo "=== ${tag} | train=${TRAIN_ROOMS} test=${TEST_ROOMS} repeat=${REPEAT} kd=${kd} fs=${fs} init=${init} standardize=${std}"
    python scripts/run_few_shot_hwild_to_wimans.py "${args[@]}" > "${log_dir}/run.log" 2>&1
    ## the runner writes its JSON/TXT next to every other run; move this run's pair into the combo dir
    ## so the sweep tree is self-contained. The path is read back from the run's own log, so parallel
    ## combos never race for the same file.
    local json txt
    json=$(sed -n 's/^Results saved to:[[:space:]]*//p' "${log_dir}/run.log" | tail -n 1)
    txt=$(sed -n 's/^Report saved to:[[:space:]]*//p' "${log_dir}/run.log" | tail -n 1)
    if [[ -n "${json}" && -f "${json}" ]]; then mv "${json}" "${log_dir}/"; fi
    if [[ -n "${txt}" && -f "${txt}" ]]; then mv "${txt}" "${log_dir}/"; fi
    return 0
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

echo "Few-shot sweep done: ${#combos[@]} combos, ${var_fail} failed (see each ${OUT_ROOT}/<tag>/run.log)"
echo "Rank them with:  python scripts/sweep_collect_few_shot.py --dir ${OUT_ROOT}"
