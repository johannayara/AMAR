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


python scripts/run_few_shot_ntu_to_wimans.py --activities walk_nothing \
  --few_shot_ratio 0.1 --kd_weight 0.1 --repeat 3 \
  --teacher_epochs 200 --student_epochs 200 \
  > "./output/few_shot_ntu_wimans/har/run_walk_nothing_${var_stamp}.log" 2>&1

## Same protocol on the union of both vocabularies: the teacher sees NTU-Fi HAR's 6 activities, the
## target test spans all 9 WiMANS activities.
python scripts/run_few_shot_ntu_to_wimans.py --activities union \
  --few_shot_ratio 0.1 --kd_weight 0.1 --repeat 3 \
  --teacher_epochs 200 --student_epochs 200 \
  > "./output/few_shot_ntu_wimans/har/run_union_${var_stamp}.log" 2>&1

end=$(date +%s)

echo "Total runtime: $((end - start)) seconds"
