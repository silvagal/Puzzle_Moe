# Puzzle-MoE: Neuro-Symbolic Mixture of Experts for ECG Diagnosis

## Overview
This repository provides the reference implementation for the paper
*Puzzle-MoE: A Neuro-Symbolic Expert Architecture for Interpretable and Rare-Class-Aware ECG Diagnosis*. The code supports the two-stage pipeline described in the manuscript:
1. **ECGWavePuzzle self-supervised pretraining** (Stage I), which learns morphology-aware representations from semantic ECG patches.
2. **Neuro-symbolic Mixture-of-Experts fine-tuning** (Stage II), which combines neural embeddings and symbolic proxy features in a load-balanced router for interpretable specialization.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Data
Download PTB-XL from PhysioNet and update the dataset path in the relevant config files:
- PTB-XL: https://physionet.org/content/ptb-xl/1.0.3/

The pipeline expects the processed PTB-XL split layout used by `src/ptbxl_dataset.py`.
Update the `dataset.path` field in the YAML configs you run so that the
training and evaluation scripts can locate the data.

## End-to-End Reproduction Guide
The steps below mirror the paper experiments. Each subsection explains which
script to run, what it does, and which results it generates.

### 0) Download and Prepare PTB-XL
**Goal:** download PTB-XL and create the `records.npz` files required by the
training pipeline.

```bash
python scripts/00_prepare_ptbxl_data.py \
  --download \
  --raw_dir data/raw/ptbxl \
  --output_dir data/processed/ptbxl
```

What happens:
- Downloads the PTB-XL archive from PhysioNet (if `--download` is provided).
- Extracts the WFDB records and metadata files.
- Builds train/val/test splits using the official `strat_fold` protocol.
- Saves `records.npz` (signals, labels, patient IDs) under:
  `data/processed/ptbxl/{train,val,test}/records.npz`.

If PhysioNet returns a 403/404, download the ZIP manually and re-run without
`--download`, or pass the direct URL with `--url`:

```bash
python scripts/00_prepare_ptbxl_data.py \
  --url https://physionet.org/files/ptb-xl/1.0.3/ptb-xl-1.0.3.zip \
  --raw_dir data/raw/ptbxl \
  --output_dir data/processed/ptbxl \
  --download
```

If the direct file URL returns 404, use the PhysioNet download endpoint:

```bash
python scripts/00_prepare_ptbxl_data.py \
  --url "https://physionet.org/download/ptb-xl/1.0.3/ptb-xl-1.0.3.zip?filename=ptb-xl-1.0.3.zip" \
  --raw_dir data/raw/ptbxl \
  --output_dir data/processed/ptbxl \
  --download
```

If the ZIP URLs are still unavailable, use the wfdb downloader (recommended
fallback that pulls the dataset file-by-file from PhysioNet):

```bash
python scripts/00_prepare_ptbxl_data.py \
  --download \
  --download_method wfdb \
  --raw_dir data/raw/ptbxl \
  --output_dir data/processed/ptbxl
```

Next, set `dataset.path` in your configs to `data/processed/ptbxl`.

### 1) Stage I: ECGWavePuzzle Self-Supervised Pretraining
**Goal:** learn morphology-aware representations from semantic P/QRS/T patches.
The script runs the SSL trainer defined in `src/trainer.py` with the Stage I
configuration.

```bash
bash scripts/05_run_ssl_pretraining.sh
```

What happens:
- Loads the Stage I configuration from `configs/stage1_ssl.yaml`.
- Builds the Puzzle-MoE encoder and trains it on the ECGWavePuzzle task.
- Writes checkpoints to `checkpoints/stage1_ssl/`.

Expected output:
- SSL training logs in the console.
- Stage I checkpoints saved under `checkpoints/stage1_ssl/`.

### 2) Stage II: Puzzle-MoE Fine-Tuning (Main Results)
**Goal:** fine-tune the MoE with symbolic routing for PTB-XL multi-label
classification.

```bash
python train_moe.py --config configs/stage2_moe_resnet101.yaml
```

What happens:
- Loads the ResNet-101 MoE configuration used for the main paper results.
- Initializes the encoder (optionally from Stage I) and trains the MoE experts.
- Saves checkpoints in the directory defined by the config.

If you want to reproduce variants reported in the paper, run the same command
with one of the following configs:
- `configs/stage2_moe_resnet101_medium.yaml` (MED experts)
- `configs/stage2_moe_resnet101_large.yaml` (LAR experts)
- `configs/stage2_moe_resnet101_ultra.yaml` (ULAR experts)
- `configs/stage2_moe_resnet101_ultra_noattn.yaml` (No-Attn ablation)
- `configs/stage2_moe_resnet101_ultra_sparse.yaml` (Sparse Top-1)
- `configs/stage2_moe_resnet101_ultra_sparse_top2.yaml` (Sparse Top-2)
- `configs/stage2_moe_resnet101_ultra_swiglu.yaml` (SwiGLU experts)
- `configs/stage2_moe_resnet101_xlar_lb0.yaml` (No load balancing)
- `configs/stage2_moe_resnet101_xlar_lb0005.yaml` (LB sweep)
- `configs/stage2_moe_resnet101_xlar_lb001.yaml` (LB sweep)
- `configs/stage2_moe_resnet101_lar_no_ssl.yaml` (No SSL)
- `configs/stage2_moe_resnet101_xlar_no_ssl.yaml` (No SSL)
- `configs/stage2_moe_resnet101_ular_no_ssl.yaml` (No SSL)
- `configs/stage2_moe_resnet101_large_supervised_tuler.yaml` (Supervised gate)
- `configs/stage2_moe_resnet101_ultra_nosym_lb0.yaml` (No symbolic, no LB)

### 3) Evaluate the ResNet-101 MoE on the Test Set
**Goal:** compute AUROC metrics for a trained MoE model using the same
architecture as in `train_moe.py`.

```bash
python eval_moe_resnet101.py \
  --config configs/stage2_moe_resnet101.yaml \
  --checkpoint /path/to/moe_checkpoint.pt
```

What happens:
- Loads the MoE model with the configuration provided.
- Evaluates on the PTB-XL test split and reports per-class and macro AUROC.

### 4) Baseline Evaluation (Ribeiro and Strodthoff)
**Goal:** reproduce the baseline results for the two reference CNNs.

```bash
python eval_baseline.py --config configs/baseline_ribeiro.yaml --checkpoint /path/to/checkpoint.pt
python eval_baseline.py --config configs/baseline_strodthoff.yaml --checkpoint /path/to/checkpoint.pt
```

What happens:
- Loads the PTB-XL test split using the same preprocessing as the MoE runs.
- Evaluates the baseline model checkpoint and prints AUROC, accuracy, and F1.

### 5) Ablation Studies (R1--R3 scripts)
**Goal:** reproduce the focused ablations used in the paper discussion.

```bash
bash scripts/01_run_rq1_data_efficiency.sh
bash scripts/02_run_rq2_expert_collapse.sh
bash scripts/03_run_rq3_symbolic_tradeoff.sh
```

What happens:
- **RQ1 (data efficiency)** runs Stage I and Stage II with data subset ratios.
- **RQ2 (expert collapse)** sweeps symbolic consistency without load balancing.
- **RQ3 (symbolic trade-off)** sweeps symbolic loss weights for the MoE.

### 6) Interpretability Figures
**Goal:** generate routing, attention, and noise-probe figures for the paper.

```bash
python scripts/investigate_moe.py \
  --checkpoint checkpoints/stage2_moe/best_model.pt \
  --data_path /path/to/ptbxl/processed \
  --split test \
  --output_dir outputs/figs
```

What happens:
- Loads the MoE checkpoint and computes attention maps, gating distributions,
  and routing behavior under noise.
- Saves plots under the specified output directory.

## Repository Structure
- `src/`: Core model, dataset, and training utilities.
- `configs/`: YAML configurations used in the paper experiments.
- `scripts/`: Reproducibility scripts for the reported studies.
- `train_ssl.py`, `train_moe.py`: Main entry points for Stage I and Stage II.
- `eval_moe_resnet101.py`, `eval_baseline.py`: Evaluation utilities.

## Citation
```bibtex
@article{puzzlemoe2025,
  title={Puzzle-MoE: A Neuro-Symbolic Expert Architecture for Interpretable and Rare-Class-Aware ECG Diagnosis},
  author={Anonymous Authors},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025}
}
```

## License
Released for academic research use. See `LICENSE` for details.
