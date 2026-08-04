#!/usr/bin/env bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/ubuntu/SBI4atmret

for i in $(seq 62 82); do
  python scripts/generate_dataset.py \
    --config experiments/config_MiriGeminiHST_cloudfree_aws.yaml \
    --output-dir ../simulations/cloudfree \
    --batch-size 4096 \
    --array-index "$i" &
done

wait
