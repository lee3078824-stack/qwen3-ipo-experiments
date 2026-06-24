#!/usr/bin/env bash
set -euo pipefail

# Length-normalized target-ratio IPO experiment runner, created 2026-06-07.
#
# Examples:
#   ./run_20260607_length_normalized_ipo_experiments.sh
#   LENGTH_NORMS="mean sqrt" RATIOS="0.6 0.7 0.8" ./run_20260607_length_normalized_ipo_experiments.sh
#   OBJECTIVES="target_ratio standard_ipo" LENGTH_NORMS="sqrt" RATIOS="0.7 0.8" ./run_20260607_length_normalized_ipo_experiments.sh
#   OBJECTIVES="standard_ipo" LENGTH_NORMS="sqrt mean" ./run_20260607_length_normalized_ipo_experiments.sh
#   SUBSET=512 EVAL_SUBSET=128 REPORT_TO=none RATIOS="0.7" ./run_20260607_length_normalized_ipo_experiments.sh
#   CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=hl-tp-qwen3-preference ./run_20260607_length_normalized_ipo_experiments.sh

MODEL_ID="${MODEL_ID:-AIPlans/Qwen3-0.6b-SFT-hs2}"
DATASET_ID="${DATASET_ID:-Jennny/helpsteer2-helpfulness-preference}"

LENGTH_NORMS="${LENGTH_NORMS:-mean}"
OBJECTIVES="${OBJECTIVES:-target_ratio}"
RATIOS="${RATIOS:-0.6 0.7 0.8}"
SUBSET="${SUBSET:-0}"
EVAL_SUBSET="${EVAL_SUBSET:-0}"
EPOCHS="${EPOCHS:-3}"
BETA="${BETA:-0.01}"
LR="${LR:-5e-7}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_LENGTH="${MAX_LENGTH:-1024}"

LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"

PRECOMPUTE_REF_LOG_PROBS="${PRECOMPUTE_REF_LOG_PROBS:-true}"
AUTO_RESUME="${AUTO_RESUME:-true}"
SKIP_FINISHED="${SKIP_FINISHED:-true}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-hl-tp-qwen3-preference}"
export WANDB_PROJECT
WANDB_DIR="${WANDB_DIR:-${PWD}/.wandb_runs}"
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_DIR}/config}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_DIR}/cache}"
export WANDB_DIR
export WANDB_CONFIG_DIR
export WANDB_CACHE_DIR
HF_HOME="${HF_HOME:-${PWD}/.hf_cache}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export HF_HOME
export HF_DATASETS_CACHE
export PYTHONDONTWRITEBYTECODE

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/length_normalized_ipo/20260607}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi
TRAIN_SCRIPT="${TRAIN_SCRIPT:-src/train_length_normalized_ipo.py}"

echo "Length-normalized target-ratio IPO experiments"
echo "  date:         2026-06-07"
echo "  model:        ${MODEL_ID}"
echo "  dataset:      ${DATASET_ID}"
echo "  length_norms: ${LENGTH_NORMS}"
echo "  objectives:   ${OBJECTIVES}"
echo "  ratios:       ${RATIOS}"
echo "  subset:       ${SUBSET} (0 = full train split)"
echo "  eval_subset:  ${EVAL_SUBSET} (0 = full validation split)"
echo "  epochs:       ${EPOCHS}"
echo "  beta:         ${BETA}"
echo "  lr:           ${LR}"
echo "  batch:        ${TRAIN_BATCH_SIZE}"
echo "  effective:    chosen/rejected concat batch $((TRAIN_BATCH_SIZE * 2))"
echo "  precompute:   ${PRECOMPUTE_REF_LOG_PROBS}"
echo "  auto_resume:  ${AUTO_RESUME}"
echo "  skip_finished:${SKIP_FINISHED}"
echo "  report_to:    ${REPORT_TO}"
echo "  wandb_project:${WANDB_PROJECT}"
echo "  wandb_dir:    ${WANDB_DIR}"
echo "  hf_home:      ${HF_HOME}"
echo "  output:       ${OUTPUT_ROOT}"
echo

if [[ "${PRECOMPUTE_REF_LOG_PROBS}" == "true" ]]; then
  PRECOMPUTE_ARG="--precompute_ref_log_probs"
else
  PRECOMPUTE_ARG="--no-precompute_ref_log_probs"
fi

for OBJECTIVE in ${OBJECTIVES}; do
  if [[ "${OBJECTIVE}" != "target_ratio" && "${OBJECTIVE}" != "standard_ipo" ]]; then
    echo "Unsupported OBJECTIVE=${OBJECTIVE}; use one of: target_ratio standard_ipo" >&2
    exit 1
  fi

  for LENGTH_NORM in ${LENGTH_NORMS}; do
    if [[ "${LENGTH_NORM}" != "mean" && "${LENGTH_NORM}" != "sqrt" && "${LENGTH_NORM}" != "none" ]]; then
      echo "Unsupported LENGTH_NORM=${LENGTH_NORM}; use one of: none mean sqrt" >&2
      exit 1
    fi

    if [[ "${OBJECTIVE}" == "target_ratio" ]]; then
      TARGET_RATIO_LIST="${RATIOS}"
    else
      TARGET_RATIO_LIST="standard"
    fi

    for TARGET_RATIO in ${TARGET_RATIO_LIST}; do
      if [[ "${OBJECTIVE}" == "target_ratio" ]]; then
        RUN_NAME="objective_target_ratio_norm_${LENGTH_NORM}_ratio_${TARGET_RATIO}_e${EPOCHS}_bs${TRAIN_BATCH_SIZE}_beta${BETA}_lr${LR}"
        TARGET_RATIO_ARGS=(--target_ratio "${TARGET_RATIO}")
      else
        RUN_NAME="objective_standard_ipo_norm_${LENGTH_NORM}_e${EPOCHS}_bs${TRAIN_BATCH_SIZE}_beta${BETA}_lr${LR}"
        TARGET_RATIO_ARGS=()
      fi

      OUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
      RESUME_ARGS=()

      if [[ "${SKIP_FINISHED}" == "true" && -d "${OUT_DIR}/final" ]]; then
        echo "============================================================"
        echo "Skipping objective=${OBJECTIVE}, length_norm=${LENGTH_NORM}, target=${TARGET_RATIO}; final weights already exist."
        echo "Found: ${OUT_DIR}/final"
        echo "============================================================"
        echo
        continue
      fi

      if [[ "${AUTO_RESUME}" == "true" ]]; then
        LATEST_CHECKPOINT="$(find "${OUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)"
        if [[ -n "${LATEST_CHECKPOINT}" ]]; then
          RESUME_ARGS=(--resume_from_checkpoint "${LATEST_CHECKPOINT}")
        fi
      fi

      echo "============================================================"
      echo "Running objective=${OBJECTIVE}, length_norm=${LENGTH_NORM}, target=${TARGET_RATIO}"
      echo "Saving weights to ${OUT_DIR}/final"
      if [[ "${#RESUME_ARGS[@]}" -gt 0 ]]; then
        echo "Resuming from ${RESUME_ARGS[1]}"
      fi
      echo "============================================================"

      "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
        --model_id "${MODEL_ID}" \
        --dataset_id "${DATASET_ID}" \
        --objective "${OBJECTIVE}" \
        --length_norm "${LENGTH_NORM}" \
        "${TARGET_RATIO_ARGS[@]}" \
        --subset "${SUBSET}" \
        --eval_subset "${EVAL_SUBSET}" \
        --epochs "${EPOCHS}" \
        --beta "${BETA}" \
        --lr "${LR}" \
        --train_batch_size "${TRAIN_BATCH_SIZE}" \
        --eval_batch_size "${EVAL_BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --max_length "${MAX_LENGTH}" \
        --logging_steps "${LOGGING_STEPS}" \
        --save_steps "${SAVE_STEPS}" \
        --eval_steps "${EVAL_STEPS}" \
        --save_total_limit "${SAVE_TOTAL_LIMIT}" \
        --report_to "${REPORT_TO}" \
        --run_name "${RUN_NAME}" \
        ${PRECOMPUTE_ARG} \
        "${RESUME_ARGS[@]}" \
        --output_dir "${OUT_DIR}"

      if [[ ! -d "${OUT_DIR}/final" ]]; then
        echo "Expected saved weights at ${OUT_DIR}/final, but the directory was not found." >&2
        exit 1
      fi

      echo "Saved weights: ${OUT_DIR}/final"
      echo
    done
  done
done

echo "All length-normalized IPO experiments finished."
