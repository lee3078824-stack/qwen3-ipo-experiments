#!/usr/bin/env bash
set -euo pipefail

# Qwen3-1.7B comparison: standard IPO vs fixed target-ratio IPO (ratio=0.6).
# This uses the existing length-normalized trainer with LENGTH_NORMS=none,
# so the objectives are unnormalized standard IPO and unnormalized target-ratio IPO.

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-1.7B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/qwen3_1_7b_ipo_vs_ratio06/20260610}"

OBJECTIVES="${OBJECTIVES:-standard_ipo target_ratio}"
LENGTH_NORMS="${LENGTH_NORMS:-none}"
RATIOS="${RATIOS:-0.6}"
EPOCHS="${EPOCHS:-3}"

# 1.7B full fine-tuning needs a smaller per-device batch than the 0.6B runs.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"

MAX_LENGTH="${MAX_LENGTH:-1024}"
BETA="${BETA:-0.01}"
LR="${LR:-5e-7}"

REPORT_TO="${REPORT_TO:-none}"
WANDB_PROJECT="${WANDB_PROJECT:-hl-tp-qwen3-preference}"
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
SKIP_FINISHED="${SKIP_FINISHED:-true}"
AUTO_RESUME="${AUTO_RESUME:-true}"
PRECOMPUTE_REF_LOG_PROBS="${PRECOMPUTE_REF_LOG_PROBS:-true}"

export MODEL_ID
export OUTPUT_ROOT
export OBJECTIVES
export LENGTH_NORMS
export RATIOS
export EPOCHS
export TRAIN_BATCH_SIZE
export EVAL_BATCH_SIZE
export GRADIENT_ACCUMULATION_STEPS
export MAX_LENGTH
export BETA
export LR
export REPORT_TO
export WANDB_PROJECT
export SAVE_STEPS
export EVAL_STEPS
export SAVE_TOTAL_LIMIT
export SKIP_FINISHED
export AUTO_RESUME
export PRECOMPUTE_REF_LOG_PROBS

exec ./run_20260607_length_normalized_ipo_experiments.sh
