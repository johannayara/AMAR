#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 32G
#SBATCH --qos normal
#SBATCH --time 5:00:00
#SBATCH --gres gpu:a100:1
#SBATCH --array=0-2

set -euo pipefail
source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR

envs=(empty_room meeting_room classroom)
env=${envs[$SLURM_ARRAY_TASK_ID]}

start=$(date +%s)
WANDB_MODE=offline python scripts/run_cross_domain.py \
    --model AMAR_WO_RVQ --task location --repeat 5 --env "$env" \
    > "./output/cd/AMAR_WO_RVQ_r5_${env}_cd.txt" 2>&1
echo "Total runtime $env: $(( $(date +%s) - start )) seconds"