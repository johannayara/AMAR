#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 40G
#SBATCH --qos normal
#SBATCH --time 8:00:00
#SBATCH --gres gpu:a100:1
#SBATCH --array=0-2

#
## Few-shot knowledge distillation for the density-map group-counting model: train on one room and
## test on the other two. One array task per training room. The teacher is trained on the full
## training room, the student on a few-shot fraction of it with occupancy distillation; each test
## room's map is rendered with that room's own layout kernels.
#
set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR

envs=(empty_room meeting_room classroom)
env=${envs[$SLURM_ARRAY_TASK_ID]}

mkdir -p ./output/few_shot_density

start=$(date +%s)
WANDB_MODE=offline python scripts/run_few_shot.py \
    --model density_map --task location --repeat 3 --env "$env" \
    --few_shot_ratio 0.05 --epochs 200 --kd_weight 1.0 \
    > "./output/few_shot_density/density_map_${env}_fewshot.txt" 2>&1
echo "Total runtime $env: $(( $(date +%s) - start )) seconds"
