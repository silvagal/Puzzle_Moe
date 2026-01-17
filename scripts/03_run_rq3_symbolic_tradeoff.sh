#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/ablation.yaml

echo "Running symbolic trade-off ablation"
for lambda_sym in 0.0 0.1 1.0 10.0; do
  echo "lambda_sym=${lambda_sym}"
  python -m src.trainer --config ${CONFIG} --stage moe --lambda_sym ${lambda_sym}
done
