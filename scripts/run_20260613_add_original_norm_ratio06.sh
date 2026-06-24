#!/usr/bin/env bash
set -euo pipefail

# Add the missing ratio=0.6 runs to the original length-normalized IPO experiment.
#
# This matches the completed 2026-06-07 norm setup:
#   model:      AIPlans/Qwen3-0.6b-SFT-hs2
#   objective:  target_ratio
#   norms:      mean, sqrt
#   ratio:      0.6
#   beta:       0.01
#   lr:         5e-7
#   epochs:     3
#
# Runs:
#   1. target_ratio + mean + ratio=0.6
#   2. target_ratio + sqrt + ratio=0.6

MODEL_ID="${MODEL_ID:-AIPlans/Qwen3-0.6b-SFT-hs2}" \
DATASET_ID="${DATASET_ID:-Jennny/helpsteer2-helpfulness-preference}" \
OBJECTIVES="${OBJECTIVES:-target_ratio}" \
LENGTH_NORMS="${LENGTH_NORMS:-mean sqrt}" \
RATIOS="${RATIOS:-0.6}" \
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/length_normalized_ipo/20260607}" \
EPOCHS="${EPOCHS:-3}" \
BETA="${BETA:-0.01}" \
LR="${LR:-5e-7}" \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}" \
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}" \
MAX_LENGTH="${MAX_LENGTH:-1024}" \
SUBSET="${SUBSET:-0}" \
EVAL_SUBSET="${EVAL_SUBSET:-0}" \
LOGGING_STEPS="${LOGGING_STEPS:-10}" \
SAVE_STEPS="${SAVE_STEPS:-100}" \
EVAL_STEPS="${EVAL_STEPS:-100}" \
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}" \
REPORT_TO="${REPORT_TO:-none}" \
SKIP_FINISHED="${SKIP_FINISHED:-true}" \
AUTO_RESUME="${AUTO_RESUME:-true}" \
PRECOMPUTE_REF_LOG_PROBS="${PRECOMPUTE_REF_LOG_PROBS:-true}" \
./run_20260607_length_normalized_ipo_experiments.sh
