#!/usr/bin/env bash
set -euo pipefail

# Fixed target-ratio IPO experiment runner matching the 05/28 setup.
#
# Examples:
#   ./run_target_ratio_ipo_experiments.sh
#   RATIOS="0.6 0.7 0.8" ./run_target_ratio_ipo_experiments.sh
#   REPORT_TO=none RATIOS="0.7" ./run_target_ratio_ipo_experiments.sh
#   CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=hl-tp-qwen3-preference ./run_target_ratio_ipo_experiments.sh

MODEL_ID="${MODEL_ID:-AIPlans/Qwen3-0.6b-SFT-hs2}"
DATASET_ID="${DATASET_ID:-Jennny/helpsteer2-helpfulness-preference}"

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

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/fixed_ratio_ipo}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-src/train_target_ratio_ipo.py}"

echo "Fixed target-ratio IPO experiments"
echo "  model:        ${MODEL_ID}"
echo "  dataset:      ${DATASET_ID}"
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
echo "  output:       ${OUTPUT_ROOT}"
echo

if [[ "${PRECOMPUTE_REF_LOG_PROBS}" == "true" ]]; then
  PRECOMPUTE_ARG="--precompute_ref_log_probs"
else
  PRECOMPUTE_ARG="--no-precompute_ref_log_probs"
fi

for TARGET_RATIO in ${RATIOS}; do
  RUN_NAME="ratio_${TARGET_RATIO}_e${EPOCHS}_bs${TRAIN_BATCH_SIZE}_beta${BETA}_lr${LR}"
  OUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
  RESUME_ARGS=()

  if [[ "${SKIP_FINISHED}" == "true" && -d "${OUT_DIR}/final" ]]; then
    echo "============================================================"
    echo "Skipping target_ratio=${TARGET_RATIO}; final weights already exist."
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
  echo "Running target_ratio=${TARGET_RATIO}"
  echo "Saving weights to ${OUT_DIR}/final"
  if [[ "${#RESUME_ARGS[@]}" -gt 0 ]]; then
    echo "Resuming from ${RESUME_ARGS[1]}"
  fi
  echo "============================================================"

  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --model_id "${MODEL_ID}" \
    --dataset_id "${DATASET_ID}" \
    --target_ratio "${TARGET_RATIO}" \
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

echo "All target-ratio IPO experiments finished."
