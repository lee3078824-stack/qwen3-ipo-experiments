#!/usr/bin/env bash
set -euo pipefail

# Recommended follow-up for length-normalized IPO on Qwen3-0.6B.
#
# Why this sweep:
#   Length normalization shrinks the preference gap scale.
#   The previous beta=0.01 setting likely under-scaled the normalized gap,
#   so this run tests larger beta values with the softer sqrt normalization.
#
# Default experiments:
#   target_ratio + sqrt norm + ratio=0.6 + beta in {0.02, 0.05, 0.1}
#
# Examples:
#   ./run_20260613_qwen06b_sqrt_norm_beta_sweep.sh
#   RATIOS="0.6" BETAS="0.02 0.05" ./run_20260613_qwen06b_sqrt_norm_beta_sweep.sh
#   SUBSET=1024 EVAL_SUBSET=256 ./run_20260613_qwen06b_sqrt_norm_beta_sweep.sh
#   OBJECTIVES="target_ratio standard_ipo" RATIOS="0.6" BETAS="0.02 0.05" ./run_20260613_qwen06b_sqrt_norm_beta_sweep.sh

MODEL_ID="${MODEL_ID:-AIPlans/Qwen3-0.6b-SFT-hs2}"
DATASET_ID="${DATASET_ID:-Jennny/helpsteer2-helpfulness-preference}"

BETAS="${BETAS:-0.02 0.05 0.1}"
RATIOS="${RATIOS:-0.6}"
LENGTH_NORMS="${LENGTH_NORMS:-sqrt}"
OBJECTIVES="${OBJECTIVES:-target_ratio}"

OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-outputs/qwen06b_sqrt_norm_beta_sweep/20260613}"

EPOCHS="${EPOCHS:-3}"
LR="${LR:-5e-7}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_LENGTH="${MAX_LENGTH:-1024}"

SUBSET="${SUBSET:-0}"
EVAL_SUBSET="${EVAL_SUBSET:-0}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"

REPORT_TO="${REPORT_TO:-none}"
SKIP_FINISHED="${SKIP_FINISHED:-true}"
AUTO_RESUME="${AUTO_RESUME:-true}"
PRECOMPUTE_REF_LOG_PROBS="${PRECOMPUTE_REF_LOG_PROBS:-true}"

echo "Qwen3-0.6B sqrt-normalized IPO beta sweep"
echo "  model:        ${MODEL_ID}"
echo "  dataset:      ${DATASET_ID}"
echo "  objectives:   ${OBJECTIVES}"
echo "  length_norms: ${LENGTH_NORMS}"
echo "  ratios:       ${RATIOS}"
echo "  betas:        ${BETAS}"
echo "  epochs:       ${EPOCHS}"
echo "  lr:           ${LR}"
echo "  batch:        ${TRAIN_BATCH_SIZE}"
echo "  report_to:    ${REPORT_TO}"
echo "  output base:  ${OUTPUT_ROOT_BASE}"
echo

for BETA_VALUE in ${BETAS}; do
  echo "============================================================"
  echo "Running beta=${BETA_VALUE}"
  echo "Output root: ${OUTPUT_ROOT_BASE}/beta_${BETA_VALUE}"
  echo "============================================================"

  MODEL_ID="${MODEL_ID}" \
  DATASET_ID="${DATASET_ID}" \
  OBJECTIVES="${OBJECTIVES}" \
  LENGTH_NORMS="${LENGTH_NORMS}" \
  RATIOS="${RATIOS}" \
  OUTPUT_ROOT="${OUTPUT_ROOT_BASE}/beta_${BETA_VALUE}" \
  EPOCHS="${EPOCHS}" \
  BETA="${BETA_VALUE}" \
  LR="${LR}" \
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}" \
  MAX_LENGTH="${MAX_LENGTH}" \
  SUBSET="${SUBSET}" \
  EVAL_SUBSET="${EVAL_SUBSET}" \
  LOGGING_STEPS="${LOGGING_STEPS}" \
  SAVE_STEPS="${SAVE_STEPS}" \
  EVAL_STEPS="${EVAL_STEPS}" \
  SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" \
  REPORT_TO="${REPORT_TO}" \
  SKIP_FINISHED="${SKIP_FINISHED}" \
  AUTO_RESUME="${AUTO_RESUME}" \
  PRECOMPUTE_REF_LOG_PROBS="${PRECOMPUTE_REF_LOG_PROBS}" \
  ./run_20260607_length_normalized_ipo_experiments.sh
done

echo "All Qwen3-0.6B sqrt-normalized beta sweep experiments finished."
