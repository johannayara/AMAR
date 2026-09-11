#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 32G
#SBATCH --qos normal
#SBATCH --time 5:00:00
#SBATCH --gres gpu:a100:2

# Fail fast on errors
set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR
start=$(date +%s)
bash -c 'WANDB_MODE=offline python scripts/run_main.py --model AMAR_WO_RVQ --task location --repeat 3 --env empty_room > ./output/res_AMAR_WO_RVQ_empty.txt 2>&1'
end=$(date +%s)
echo "Total runtime empty room: $((end - start)) seconds"
start_1=$(date +%s)
bash -c 'WANDB_MODE=offline python scripts/run_main.py --model AMAR_WO_RVQ --task location --repeat 3 --env meeting_room > ./output/res_AMAR_WO_RVQ_meeting.txt 2>&1'
end=$(date +%s)
echo "Total runtime meeting room: $((end - start_1)) seconds"
start_2=$(date +%s)
bash -c 'WANDB_MODE=offline python scripts/run_main.py --model AMAR_WO_RVQ --task location --repeat 3 --env classroom > ./output/res_AMAR_WO_RVQ_classroom.txt 2>&1'
end=$(date +%s)
echo "Total runtime classroom: $((end - start_2)) seconds"
echo "Total runtime: $((end - start)) seconds"