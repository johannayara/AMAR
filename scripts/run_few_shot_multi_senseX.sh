#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 40G
#SBATCH --qos normal
#SBATCH --time 8:00:00
#SBATCH --gres gpu:a100:1

set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR
env=empty_room
mkdir -p "./output/few_shot/${env}"

start=$(date +%s)
echo "Start: $(date)"
export WANDB_MODE=offline
python scripts/run_few_shot_multi_senseX.py --model multiSense_X --repeat 3 --env "${env}" \
  --few_shot_ratio 0.05 --epochs 200 --kd_weight 1.0 > "./output/few_shot/${env}/multiSenseX_fewshot.txt" 2>&1
end=$(date +%s)
echo "Total runtime: $((end - start)) seconds"
