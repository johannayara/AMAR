#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 16G
#SBATCH --qos normal
#SBATCH --time 4:00:00
#SBATCH --gres gpu:a100:1

set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR

mkdir -p ./output/single

start=$(date +%s)
echo "Start: $(date)"
export WANDB_MODE=offline 
python scripts/run_main.py --model AMAR_WO_RVQ --task location --repeat 3 --env empty_room > "./output/single/AMAR_WO_RVQ_test.txt" 2>&1
end=$(date +%s)
echo "Total runtime: $((end - start)) seconds"