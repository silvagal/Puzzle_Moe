#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/stage2_moe.yaml

echo "Running expert collapse analysis"
for lambda_sym in 0.0 0.1; do
  echo "lambda_sym=${lambda_sym}"
  python -m src.trainer --config ${CONFIG} --stage moe --lambda_sym ${lambda_sym}
done
