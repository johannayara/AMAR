#!/usr/bin/env bash
#SBATCH --job-name=density_threshold_sweep
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --qos normal
#SBATCH --time 12:00:00
#SBATCH --gres gpu:a100:2
#SBATCH --array=0-2
#SBATCH --output=slurm_density_threshold_sweep_%A_%a.out

set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Keep only the envs you really need
envs=(empty_room meeting_room classroom)
env="${envs[${SLURM_ARRAY_TASK_ID:-0}]}"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

# Override from sbatch, e.g.:
#   EMPTY_THRESHOLDS="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9" sbatch scripts/sweep_density.sh
EMPTY_THRESHOLDS="${EMPTY_THRESHOLDS:-"0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90"}"

best_threshold=""
best_score=""
best_out=""

for thr in $EMPTY_THRESHOLDS; do
    out="output/sweep/density_map_${env}/empty_threshold_${thr}"
    mkdir -p "$out"

    echo "=== Running ${env} with empty_threshold=${thr} ==="

    python scripts/sweep_density.py \
        --env "$env" \
        --out "$out" \
        --metric "${METRIC:-avg_balanced_accuracy}" \
        --tie_break "${TIE_BREAK:-avg_mae}" \
        --seed "${SEED:-39}" \
        --workers "${WORKERS:-1}" \
        --gpus "${GPUS:-0}" \
        --empty_threshold "$thr" \
        "$@" \
        > "${out}/sweep.log" 2>&1 || {
            echo "Threshold ${thr} failed; skipping."
            continue
        }

    # Try to read a metric from the generated result file(s)
    score="$(python - <<'PY' "$out"
import json, os, glob, sys

base = sys.argv[1]
candidates = []
for p in [os.path.join(base, "best.json"), os.path.join(base, "results.json"), os.path.join(base, "summary.json")]:
    if os.path.exists(p):
        candidates.append(p)

if not candidates:
    # fallback: scan json files
    candidates = glob.glob(os.path.join(base, "**", "*.json"), recursive=True)

best = None
for p in candidates:
    try:
        with open(p, "r") as f:
            data = json.load(f)
    except Exception:
        continue

    # Common patterns: {"best_score": ...}, {"metrics": {"avg_balanced_accuracy": ...}}, {"avg_balanced_accuracy": ...}
    if isinstance(data, dict):
        score = None
        if "best_score" in data:
            score = data["best_score"]
        elif "score" in data:
            score = data["score"]
        elif "metrics" in data and isinstance(data["metrics"], dict):
            score = data["metrics"].get("avg_balanced_accuracy", data["metrics"].get("accuracy"))
        else:
            score = data.get("avg_balanced_accuracy", data.get("accuracy"))

        if score is not None:
            best = float(score)
            break

if best is None:
    print("nan")
else:
    print(best)
PY
)"

    echo "empty_threshold=${thr} score=${score}"

    # Keep the best threshold by whichever metric you want.
    # Here we keep the largest score.
    if [[ "$score" != "nan" ]] && [[ -z "${best_score}" || "$(python - <<'PY' "$score" "$best_score"
import sys
try:
    a = float(sys.argv[1])
    b = float(sys.argv[2])
    print(1 if a > b else 0)
except Exception:
    print(0)
PY
)" == "1" ]]; then
        best_threshold="$thr"
        best_score="$score"
        best_out="$out"
    fi
done

echo "BEST empty_threshold=${best_threshold} with score=${best_score}"
echo "BEST OUTDIR: ${best_out}"

# Optional: print a one-line summary for easy parsing
printf '%s\n' "BEST_THRESHOLD=${best_threshold}" "BEST_SCORE=${best_score}" "BEST_OUT=${best_out}"