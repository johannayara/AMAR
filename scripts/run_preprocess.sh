#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 32G
#SBATCH --qos normal
#SBATCH --time 8:00:00
#SBATCH --gres gpu:v100:1

set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR


model=density_map
env=meeting_room

mkdir -p "./output/preprocess/"

start=$(date +%s)
echo "Start: $(date)"
export WANDB_MODE=offline 

for p in none static,temporal,zscore static zscore; do
  AMAR_CSI_PREPROCESS=$p python scripts/run_cross_domain.py --model density_map --task location \
    --repeat 3 --train_envs empty_room,classroom --epochs 70 --hp density.count_rule=layout_free > "./output/preprocess/${p}_run_$(date +%Y%m%d_%H%M).log" 2>&1
done
end=$(date +%s)

echo "Total runtime: $((end - start)) seconds"