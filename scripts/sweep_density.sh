#!/usr/bin/env bash
#SBATCH --job-name=density_threshold_sweep
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --qos normal
#SBATCH --time 12:00:00
#SBATCH --gres gpu:a100:1
#SBATCH --array=0-2
#SBATCH --output=slurm_density_threshold_sweep_%A_%a.out

#
## Calibrate `density.empty_threshold` (the empty gate, stage 1) and `density.eval_threshold` (the
## occupancy threshold, stage 2) SEPARATELY, for each count rule, one room per array task.
##
## For every rule in RULES this runs two independent 1-D sweeps:
##   * empty sweep: vary empty_threshold, hold eval_threshold at EVAL_FIXED
##   * eval  sweep: vary eval_threshold,  hold empty_threshold at EMPTY_FIXED
## Each cell is one `scripts/run_main.py` invocation with the pair pinned through `--hp`, so the cell
## is a full, isolated train+eval. run_main writes a JSON per invocation; this script reads it back.
##
## Selection is on the VALIDATION split (valid_avg_*), never on the split it is then reported on:
## calibrating on the scored split lets a value win by fitting the sweep rather than the model -- the
## trap documented next to `density.eval_threshold` in configs/preset.py.
##
## Ranking metric, by rule and knob (override with the matching environment variable):
##   template     empty -> valid_avg_balanced_accuracy   eval -> valid_avg_balanced_accuracy
##   layout_free  empty -> valid_avg_balanced_accuracy   (the gate now zeroes the peak count too)
##                eval  -> valid_avg_occupancy_f1       (eval only sets the occupancy readout here)
## Under "layout_free" the reported count is the number of map peaks; the empty gate zeroes it (and the
## mass count and the positions) on a frame whose map maximum is below the gate, but eval_threshold
## only sets the occupancy readout, so the two knobs are still ranked by what they actually move.
##
## Submit from the repository root:
##     sbatch scripts/sweep_density.sh
##     sbatch --array=0 scripts/sweep_density.sh
##     RULES=template EMPTY_THRESHOLDS="0.06 0.09 0.12" EVAL_THRESHOLDS="0.4 0.5 0.6" \
##         sbatch scripts/sweep_density.sh
##
## Environment variables:
##     RULES             space-separated count rules        (default "template layout_free")
##     EMPTY_THRESHOLDS  space-separated empty-gate grid    (default 0.04..0.24 step 0.02)
##     EVAL_THRESHOLDS   space-separated occupancy grid     (default 0.30..0.70 step 0.10)
##     EVAL_FIXED        eval held while sweeping empty     (default 0.5, neutral boundary)
##     EMPTY_FIXED       empty held while sweeping eval     (default 0.0, i.e. gate off)
##     METRIC_EMPTY      override the empty-sweep metric (max)
##     METRIC_EVAL       override the eval-sweep metric (max)
##     TIE_BREAK_EMPTY   override the empty-sweep tie-break (min)
##     TIE_BREAK_EVAL    override the eval-sweep tie-break (min)
##     REPEAT            repeats per cell                   (default 1)
##     EPOCHS            epochs per repeat                  (default 100 = preset nn.epoch)
##
## Note: run_main reserves a figure/checkpoint directory under visualizations/density_map/<backbone>/
## <env>/ per invocation, so a large grid leaves one such directory per cell. Clean that tree between
## sweeps if disk matters.
#
set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

envs=(empty_room meeting_room classroom)
env="${envs[${SLURM_ARRAY_TASK_ID:-0}]}"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

RULES="${RULES:-"template layout_free"}"
EMPTY_THRESHOLDS="${EMPTY_THRESHOLDS:-"0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20 0.22 0.24"}"
EVAL_THRESHOLDS="${EVAL_THRESHOLDS:-"0.30 0.40 0.50 0.60 0.70"}"
EVAL_FIXED="${EVAL_FIXED:-0.5}"
EMPTY_FIXED="${EMPTY_FIXED:-0.0}"
METRIC_EMPTY="${METRIC_EMPTY:-}"
METRIC_EVAL="${METRIC_EVAL:-}"
TIE_BREAK_EMPTY="${TIE_BREAK_EMPTY:-}"
TIE_BREAK_EVAL="${TIE_BREAK_EVAL:-}"
REPEAT="${REPEAT:-1}"
EPOCHS="${EPOCHS:-100}"

out_root="output/sweep/density_map_${env}/thresholds"
mkdir -p "$out_root"
index="$out_root/index.tsv"
: > "$index"

## run one cell (rule, knob, value); the other threshold is pinned at its fixed baseline
run_cell() {
    local rule="$1" knob="$2" value="$3" empty_thr eval_thr
    if [[ "$knob" == "empty" ]]; then
        empty_thr="$value"; eval_thr="$EVAL_FIXED"
    else
        empty_thr="$EMPTY_FIXED"; eval_thr="$value"
    fi
    local out="$out_root/$rule/${knob}_${value}"
    mkdir -p "$out"
    echo "=== ${env} [${rule}] ${knob}_threshold=${value} (other: empty=${empty_thr} eval=${eval_thr}) ==="

    if ! python scripts/run_main.py \
        --model density_map --task location --env "$env" \
        --repeat "$REPEAT" --epochs "$EPOCHS" \
        --hp density.count_rule="$rule" \
        --hp density.empty_threshold="$empty_thr" \
        --hp density.eval_threshold="$eval_thr" \
        --hp path.save_dir="$out" \
        > "$out/run.log" 2>&1; then
        echo "  run failed; see $out/run.log"
        return
    fi

    ## run_main writes exactly one aggregated JSON per invocation
    shopt -s nullglob
    local jsons=("$out"/single/*.json)
    if [[ ${#jsons[@]} -eq 0 ]]; then
        echo "  no result JSON under $out/single; see $out/run.log"
        return
    fi

    local parsed
    parsed="$(python - "${jsons[0]}" "$rule" "$knob" \
        "$METRIC_EMPTY" "$METRIC_EVAL" "$TIE_BREAK_EMPTY" "$TIE_BREAK_EVAL" <<'PY'
import json, sys

## validation metrics to rank on, per rule and knob. Layout-free's reported count is the peak count
## and is flat in both thresholds, so each knob is scored by the metric it actually moves.
DEFAULTS = {
    ("template", "empty"):    ("valid_avg_balanced_accuracy", "valid_avg_mae"),
    ("template", "eval"):     ("valid_avg_balanced_accuracy", "valid_avg_mae"),
    ("layout_free", "empty"): ("valid_avg_balanced_accuracy", "valid_avg_mae"),
    ("layout_free", "eval"):  ("valid_avg_occupancy_f1", "valid_avg_occupancy_accuracy"),
}

var_json, var_rule, var_knob, var_metric_empty, var_metric_eval, var_tie_empty, var_tie_eval = sys.argv[1:8]
var_metric_default, var_tie_default = DEFAULTS.get((var_rule, var_knob), DEFAULTS[("template", var_knob)])
var_metric_name = (var_metric_empty if var_knob == "empty" else var_metric_eval) or var_metric_default
var_tie_name = (var_tie_empty if var_knob == "empty" else var_tie_eval) or var_tie_default

with open(var_json) as var_file:
    var_data = json.load(var_file)


def render(var_key):
    var_value = var_data.get(var_key)
    return "nan" if var_value is None else repr(float(var_value))


print(f"{var_metric_name}\t{render(var_metric_name)}\t{var_tie_name}\t{render(var_tie_name)}")
PY
)"
    local metric_name metric_value tie_name tie_value
    IFS=$'\t' read -r metric_name metric_value tie_name tie_value <<< "$parsed"
    echo "  ${metric_name}=${metric_value} ${tie_name}=${tie_value}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$rule" "$knob" "$value" "$metric_name" "$metric_value" "$tie_name" "$tie_value" "$out" >> "$index"
}

for rule in $RULES; do
    for empty_thr in $EMPTY_THRESHOLDS; do
        run_cell "$rule" "empty" "$empty_thr"
    done
    for eval_thr in $EVAL_THRESHOLDS; do
        run_cell "$rule" "eval" "$eval_thr"
    done
done

echo
echo "=== selecting best threshold per rule and knob for ${env} ==="
python - "$index" "$out_root/summary.json" "$env" <<'PY'
import json
import math
import sys

var_index_path, var_summary_path, var_env = sys.argv[1:4]
var_groups = {}
with open(var_index_path) as var_file:
    for var_line in var_file:
        var_parts = var_line.rstrip("\n").split("\t")
        if len(var_parts) != 8:
            continue
        var_rule, var_knob, var_value, var_metric, var_score, var_tie, var_tie_score, var_out = var_parts
        try:
            var_score, var_tie_score, var_value = float(var_score), float(var_tie_score), float(var_value)
        except ValueError:
            continue
        if math.isnan(var_score):
            continue
        var_groups.setdefault((var_rule, var_knob), []).append({
            "value": var_value, "metric": var_metric, "score": var_score,
            "tie_break": var_tie, "tie_break_score": var_tie_score, "out": var_out})

if not var_groups:
    print("no successful cells; nothing to select")
    sys.exit(1)

var_summary = {"env": var_env, "sweeps": {}}
for (var_rule, var_knob), var_rows in var_groups.items():
    ## maximize the metric, minimize the tie-break
    var_best = sorted(var_rows, key=lambda var_row: (-var_row["score"], var_row["tie_break_score"]))[0]
    var_summary["sweeps"][f"{var_rule}/{var_knob}"] = {"best": var_best, "runs": var_rows}
    print(f"BEST {var_rule:12s} {var_knob}_threshold={var_best['value']:<5} "
          f"{var_best['metric']}={var_best['score']:.4f} {var_best['tie_break']}={var_best['tie_break_score']:.4f} "
          f"({var_best['out']})")

with open(var_summary_path, "w") as var_file:
    json.dump(var_summary, var_file, indent=4)
print(f"summary: {var_summary_path}")
PY
