#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/stage2_moe.yaml

echo "Running data efficiency study: subset ratios 0.01, 0.1, 1.0"
for ratio in 0.01 0.1 1.0; do
  echo "Subset ratio: ${ratio}"
  python -m src.trainer --config ${CONFIG} --stage ssl --subset_ratio ${ratio}
  python -m src.trainer --config ${CONFIG} --stage moe --subset_ratio ${ratio}
  echo "Completed ratio ${ratio}"
done
