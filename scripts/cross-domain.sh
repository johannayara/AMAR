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

model=resnet1d
mkdir -p "./output/cd/sim_wimans/${model}/"

# Train on the real rooms + anywhere, test on the held-out real room and every simulated room.
python scripts/run_cross_domain.py --model "${model}" --task location \
  --train_envs "meeting_room, empty_room, anywhere" \
  --repeat 3 --epochs 70 \
  --hp data.synth_wimans.enabled=true \
  --hp hwild.grad_clip=1.0 --hp hwild.ema_decay=0.999 \
  > "./output/cd/sim_wimans/${model}/run_$(date +%Y%m%d_%H%M).log" 2>&1

# Train on anywhere alone, test on every real and simulated room.
python scripts/run_cross_domain.py --model "${model}" --task location \
  --train_envs "anywhere" \
  --repeat 3 --epochs 70 \
  --hp data.synth_wimans.enabled=true \
  --hp hwild.grad_clip=1.0 --hp hwild.ema_decay=0.999 \
  > "./output/cd/sim_wimans/${model}/run_$(date +%Y%m%d_%H%M).log" 2>&1

end=$(date +%s)
echo "Total runtime: $((end - start)) seconds"
