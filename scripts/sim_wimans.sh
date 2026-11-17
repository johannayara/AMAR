#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 16G
#SBATCH --qos normal
#SBATCH --time 4:00:00
#SBATCH --gres gpu:a100:1

set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR

mkdir -p "./output/sim_wimans/few-shot-anywhere/normal_rooms"
mkdir -p "./output/sim_wimans/few-shot-anywhere/sim_rooms"

start=$(date +%s)
echo "Start: $(date)"
export WANDB_MODE=offline 
export AMAR_ENCODER_NORM=group

python scripts/run_few_shot_multi_rooms.py \
  --backbone resnet1d \
  --train_rooms "meeting_room, empty_room" \
  --test_rooms "classroom, meeting_room_synth, empty_room_synth, classroom_synth" \
  --support_rooms classroom --repeat 3 --standardize window \
  --teacher_epochs 70 --student_epochs 70 --few_shot_ratio 0.1 --kd_weight 0 \
  --hp data.synth_wimans.enabled=true --no_init_from_teacher \
  > "./output/sim_wimans/few-shot-anywhere/normal_rooms/run_synth_$(date +%Y%m%d_%H%M).log" 2>&1

python scripts/run_few_shot_multi_rooms.py \
  --backbone resnet1d \
  --train_rooms "meeting_room_synth, empty_room_synth, classroom_synth" \
  --test_rooms "classroom, meeting_room, empty_room" \
  --support_rooms classroom --repeat 3 --standardize window \
  --teacher_epochs 70 --student_epochs 70 --few_shot_ratio 0.1 --kd_weight 0 \
  --hp data.synth_wimans.enabled=true --no_init_from_teacher \
  > "./output/sim_wimans/few-shot-anywhere/sim_rooms/run_synth_$(date +%Y%m%d_%H%M).log" 2>&1

end=$(date +%s)

echo "Total runtime: $((end - start)) seconds"