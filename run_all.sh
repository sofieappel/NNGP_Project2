#!/bin/bash
set -e
mkdir -p /nngp/output

DATASETS=${@:-mnist fashion_mnist}   # default if no arguments are given

for ds in $DATASETS; do
  echo "=== Running dataset: $ds ==="
  python uncertainty_plot.py \
    --dataset=$ds \
    --num_train=1000 --num_eval=10000 \
    --hparams='depth=3,weight_var=2.0,bias_var=0.2' \
    --nonlinearities='tanh,relu' \
    --output_file=/nngp/output/uncertainty_${ds}.png
done
