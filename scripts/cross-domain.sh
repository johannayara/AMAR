#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 32G
#SBATCH --qos normal
#SBATCH --time 6:00:00
#SBATCH --gres gpu:a100:2

# Fail fast on errors
set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR
start=$(date +%s)
bash -c 'WANDB_MODE=offline python scripts/run_cross_domain.py --model AMAR_WO_RVQ --task location --repeat 3 --env meeting_room' > "./output/cd/AMAR_WO_RVQ_meeting_room_single.txt" 2>&1
end=$(date +%s)
echo "Total runtime: $((end - start)) seconds"