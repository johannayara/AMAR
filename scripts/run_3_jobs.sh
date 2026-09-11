#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 32G
#SBATCH --qos normal
#SBATCH --time 4:00:00
#SBATCH --gres gpu:a100:1
#SBATCH --array=0-2

set -euo pipefail
source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR

envs=(empty_room meeting_room classroom)
env=${envs[$SLURM_ARRAY_TASK_ID]}

start=$(date +%s)
WANDB_MODE=offline python scripts/run_main.py \
    --model AMAR_WO_RVQ --task location --repeat 3 --env "$env" \
    > "./output/res_AMAR_WO_RVQ_${env}.txt" 2>&1
echo "Total runtime $env: $(( $(date +%s) - start )) seconds"