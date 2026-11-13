#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 16G
#SBATCH --qos normal
#SBATCH --time 4:00:00
#SBATCH --gres gpu:v100:1

set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR

mkdir -p "./output/sim_wimans/"

start=$(date +%s)
echo "Start: $(date)"
export WANDB_MODE=offline 
export AMAR_ENCODER_NORM=group
python scripts/run_few_shot_multi_rooms.py \
  --train_rooms "meeting_room_synth, classroom_synth, empty_room_synth" \
  --test_rooms "meeting_room, empty_room" \
  --support_rooms classroom --repeat 3 \
  --teacher_epochs 70 --student_epochs 70 --few_shot_ratio 0.1 --kd_weight 0 \
  --hp data.synth_wimans.enabled=true \
  > "./output/sim_wimans/run_synth_$(date +%Y%m%d_%H%M).log" 2>&1

end=$(date +%s)

echo "Total runtime: $((end - start)) seconds"