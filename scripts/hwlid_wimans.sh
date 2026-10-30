#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 16G
#SBATCH --qos normal
#SBATCH --time 4:00:00
#SBATCH --gres gpu:v100:1

set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR

mkdir -p "./output/few_shot_hwild_wimans/group"

start=$(date +%s)
echo "Start: $(date)"
export WANDB_MODE=offline 
export AMAR_ENCODER_NORM=group
## python scripts/run_hwild_to_wimans.py --repeat 3 --epochs 200 --train_rooms "Conference, Laboratory, empty_room, classroom" --test_rooms "meeting_room, Lounge, Office" > "./output/hwild_wimans/run_$(date +%Y%m%d_%H%M).log" 2>&1
python scripts/run_few_shot_hwild_to_wimans.py \
  --train_rooms "Conference, Laboratory, Lounge, Office" --test_rooms "meeting_room, empty_room" \
  --support_rooms classroom --repeat 3 \
  --teacher_epochs 200 --student_epochs 200 --few_shot_ratio 0.1 --kd_weight 0 > "./output/few_shot_hwild_wimans/group/run_$(date +%Y%m%d_%H%M).log" 2>&1


# python scripts/run_few_shot_hwild_to_wimans.py --repeat 3 \
#   --teacher_epochs 200 --student_epochs 200 \
#   --train_rooms "Conference, Laboratory, Lounge, Office" \
#   --test_rooms "meeting_room, classroom, empty_room" \
#   --few_shot_ratio 0.1 --kd_weight 0.1 > "./output/few_shot_hwild_wimans/group/run_$(date +%Y%m%d_%H%M).log" 2>&1 

end=$(date +%s)

echo "Total runtime: $((end - start)) seconds"