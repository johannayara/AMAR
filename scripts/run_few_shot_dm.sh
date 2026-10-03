#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 40G
#SBATCH --qos normal
#SBATCH --time 5:00:00
#SBATCH --gres gpu:a100:1

#
set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR

env="empty_room"

mkdir -p ./output/few_shot_density

start=$(date +%s)
WANDB_MODE=offline python scripts/run_few_shot.py \
    --model density_map --task location --repeat 3 --env "$env" \
    --few_shot_ratio 0.05 --epochs 50 --kd_weight 1.0 \
    > "./output/few_shot_density/density_map_${env}_fewshot_1.txt" 2>&1
echo "Total runtime $env: $(( $(date +%s) - start )) seconds"