#!/bin/bash
set -euo pipefail

# Initialize conda for this shell 
source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR

# Make sure output directory exists
mkdir -p ./output

start=$(date +%s)
echo "Start: $(date)"

python scripts/run_main.py --model AMAR_WO_RVQ --task location --repeat 3 \
    > ./output/res_AMAR_WO_RVQ_run_1.txt 2>&1 &
end=$(date +%s)
echo "First runtime: $((end - start)) seconds"
python scripts/run_main.py --model AMAR --task location --repeat 3 \
    > ./output/res_AMAR_run_1.txt 2>&1 &
end=$(date +%s)
echo "Second runtime: $((end - start)) seconds"
wait

end=$(date +%s)
echo "Total runtime: $((end - start)) seconds"