#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 32G
#SBATCH --qos normal
#SBATCH --time 6:00:00
#SBATCH --gres gpu:v100:2

# Fail fast on errors
set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR

start=$(date +%s)
export WANDB_MODE=offline

# Non-few-shot reference for the few-shot resnet1d runs (scripts/sim_wimans.sh): the same backbone
# (ResNet1DLocalizer) and the same rooms -- real plus synthetic plus the few-shot "anywhere" source --
# but trained on the full training rooms, so the few-shot student/teacher can be compared against it.
# Same loading, shared frame and epoch count as the few-shot runner.
model=resnet1d
mkdir -p "./output/cd/sim_wimans/${model}/"

# Train on the real rooms + anywhere, test on the held-out real room and every simulated room.
python scripts/run_cross_domain.py --model "${model}" --task location \
  --train_envs "meeting_room, empty_room, anywhere" \
  --repeat 3 --epochs 70 \
  --hp data.synth_wimans.enabled=true \
  > "./output/cd/sim_wimans/${model}/run_$(date +%Y%m%d_%H%M).log" 2>&1

# Train on anywhere alone, test on every real and simulated room.
python scripts/run_cross_domain.py --model "${model}" --task location \
  --train_envs "anywhere" \
  --repeat 3 --epochs 70 \
  --hp data.synth_wimans.enabled=true \
  > "./output/cd/sim_wimans/${model}/run_$(date +%Y%m%d_%H%M).log" 2>&1

end=$(date +%s)
echo "Total runtime: $((end - start)) seconds"
