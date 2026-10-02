#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 16G
#SBATCH --qos normal
#SBATCH --time 4:00:00
#SBATCH --gres gpu:a100:1

set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR


model=density_map
env=classroom

mkdir -p "./output/${model}/${env}"

start=$(date +%s)
echo "Start: $(date)"
export WANDB_MODE=offline 
python scripts/run_main.py --model "${model}" --task location --repeat 3 --env "${env}" \
  > "./output/${model}/${env}.txt" 2>&1
end=$(date +%s)

echo "Total runtime: $((end - start)) seconds"