#!/usr/bin/env bash
set -euo pipefail

# Run SSL pretraining (Stage 1).

# Ensure src is on the Python path.
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# Create checkpoint directory.
mkdir -p checkpoints/stage1_ssl

python -m src.trainer \
  --config configs/stage1_ssl.yaml \
  --stage ssl
