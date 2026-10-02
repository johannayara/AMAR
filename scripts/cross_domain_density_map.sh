#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 32G
#SBATCH --qos normal
#SBATCH --time 6:00:00
#SBATCH --gres gpu:a100:1
#SBATCH --array=0-2

#
## Cross-domain density-map group counting: train on one room and test on the other two.
## One array task per training room. The occupancy threshold is calibrated on the training room's
## validation split only, and each test room's map is rendered with that room's own layout kernels.
#
set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR

envs=(empty_room meeting_room classroom)
env=${envs[$SLURM_ARRAY_TASK_ID]}

mkdir -p ./output/cd

start=$(date +%s)
WANDB_MODE=offline python scripts/run_cross_domain.py \
    --model density_map --task location --repeat 5 --env "$env" \
    > "./output/cd/density_map_${env}_cd.txt" 2>&1
echo "Total runtime $env: $(( $(date +%s) - start )) seconds"
