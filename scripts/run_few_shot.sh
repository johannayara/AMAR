#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 40G
#SBATCH --qos normal
#SBATCH --time 6:00:00
#SBATCH --gres gpu:a100:2

set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR
env=empty_room
mkdir -p "./output/few_shot/${env}"

start=$(date +%s)
echo "Start: $(date)"
export WANDB_MODE=offline
python scripts/run_few_shot.py --model AMAR_WO_RVQ --task location --repeat 3 --env "${env}" \
  --few_shot_ratio 0.01 --kd_weight 1.0 > "./output/few_shot/${env}/AMAR_WO_RVQ_fewshot.txt" 2>&1
end=$(date +%s)
echo "Total runtime: $((end - start)) seconds"
