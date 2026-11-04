#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 32G
#SBATCH --qos normal
#SBATCH --time 8:00:00
#SBATCH --gres gpu:v100:1

set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR

mkdir -p "./output/few_shot_ntu_wimans/har"

start=$(date +%s)
echo "Start: $(date)"
export WANDB_MODE=offline
export AMAR_ENCODER_NORM=group
var_stamp=$(date +%Y%m%d_%H%M)

## WiMANS occupant counts the support/query pool keeps. The default "1" gives the set-prediction head
## exactly one real target per frame, so the Hungarian matching uses one query and the head collapses
## to a single-slot classifier no matter what SetDistillationLoss does (the report prints
## "Slots 1/10 used"). Widening it gives the matching several real targets per frame, which is what
## lets the queries specialise and what makes the set-aware distillation observable.
var_wimans_users="1,2,3"


## Synth in the test set: --hp enables the second WiMANS-format source for this run, so the default
## --wimans_rooms (wimans_room_names()) becomes the three real rooms plus their "<room>_synth" keys,
## real rooms first. The support and student-validation frames stay on the real rooms only; every
## synth frame is query/test. Without --hp the source is off and no *_synth room exists.
# python scripts/run_few_shot_ntu_to_wimans.py --activities walk_nothing \
#   --few_shot_ratio 0.1 --kd_weight 0.1 --repeat 3 \
#   --teacher_epochs 200 --student_epochs 200 \
#   --wimans_users "${var_wimans_users}" \
#   --hp nn.select_metric=micro_f1 \
#   > "./output/few_shot_ntu_wimans/har/run_walk_nothing_${var_stamp}.log" 2>&1
python scripts/run_few_shot_ntu_to_wimans.py --activities walk_nothing \
  --few_shot_ratio 0.1 --kd_weight 0.1 --repeat 3 \
  --teacher_epochs 200 --student_epochs 200 \
  --wimans_users "${var_wimans_users}" \
  --hp data.synth_wimans.enabled=true \
  --hp nn.select_metric=micro_f1 \
  > "./output/few_shot_ntu_wimans/har/run_walk_nothing_synth_${var_stamp}.log" 2>&1

## Same protocol on the union of both vocabularies: the teacher sees NTU-Fi HAR's 6 activities, the
## target test spans all 9 WiMANS activities (real rooms + synth rooms).
# python scripts/run_few_shot_ntu_to_wimans.py --activities union \
#   --few_shot_ratio 0.1 --kd_weight 0.1 --repeat 3 \
#   --teacher_epochs 200 --student_epochs 200 \
#   --wimans_users "${var_wimans_users}" \
#   --hp nn.select_metric=micro_f1 \
#   > "./output/few_shot_ntu_wimans/har/run_union_${var_stamp}.log" 2>&1
python scripts/run_few_shot_ntu_to_wimans.py --activities union \
  --few_shot_ratio 0.1 --kd_weight 0.1 --repeat 3 \
  --teacher_epochs 200 --student_epochs 200 \
  --wimans_users "${var_wimans_users}" \
  --hp data.synth_wimans.enabled=true \
  --hp nn.select_metric=micro_f1 \
  > "./output/few_shot_ntu_wimans/har/run_union_synth_${var_stamp}.log" 2>&1

end=$(date +%s)

echo "Total runtime: $((end - start)) seconds"
