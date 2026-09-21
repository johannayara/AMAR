#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 40G
#SBATCH --qos normal
#SBATCH --time 7:30:00
#SBATCH --gres gpu:a100:2

set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR
env=classroom
mkdir -p "./output/t2t1/${env}"

start=$(date +%s)
echo "Start: $(date)"
WANDB_MODE=offline python scripts/run_t2t1.py --model AMAR_WO_RVQ --task location --repeat 3 --env "${env}" \
  > "./output/t2t1/${env}/AMAR_WO_RVQ_r5_300.txt" 2>&1
end=$(date +%s)
echo "Total runtime: $((end - start)) seconds"