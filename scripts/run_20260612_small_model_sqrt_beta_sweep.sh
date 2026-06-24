#!/usr/bin/env bash
set -euo pipefail

# Small-model follow-up experiment for length-normalized IPO.
# Tests whether larger beta values better match the normalized gap scale.
#
# Default sweep:
#   target_ratio + sqrt + ratio=0.7 + beta in {0.02, 0.05, 0.1}
#
# Example:
#   ./run_20260612_small_model_sqrt_beta_sweep.sh
#   BETAS="0.02 0.05" RATIOS="0.7 0.8" ./run_20260612_small_model_sqrt_beta_sweep.sh

MODEL_ID="${MODEL_ID:-AIPlans/Qwen3-0.6b-SFT-hs2}"
DATASET_ID="${DATASET_ID:-Jennny/helpsteer2-helpfulness-preference}"

BETAS="${BETAS:-0.02 0.05 0.1}"
RATIOS="${RATIOS:-0.7}"
LENGTH_NORMS="${LENGTH_NORMS:-sqrt}"
OBJECTIVES="${OBJECTIVES:-target_ratio}"

OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-outputs/length_normalized_ipo_beta_sweep/20260612}"

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

for BETA_VALUE in ${BETAS}; do
  echo "============================================================"
  echo "Starting small-model sqrt beta sweep run"
  echo "  model:        ${MODEL_ID}"
  echo "  ratios:       ${RATIOS}"
  echo "  length_norms: ${LENGTH_NORMS}"
  echo "  objectives:   ${OBJECTIVES}"
  echo "  beta:         ${BETA_VALUE}"
  echo "  output root:  ${OUTPUT_ROOT_BASE}/beta_${BETA_VALUE}"
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

echo "All small-model sqrt beta sweep experiments finished."
