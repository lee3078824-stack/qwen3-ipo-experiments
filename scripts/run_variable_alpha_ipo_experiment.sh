#!/usr/bin/env bash
set -euo pipefail

# Variable-alpha IPO experiment runner matching the 05/28 setup.
#
# Examples:
#   ./run_variable_alpha_ipo_experiment.sh
#   REPORT_TO=none ./run_variable_alpha_ipo_experiment.sh
#   CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=hl-tp-qwen3-preference ./run_variable_alpha_ipo_experiment.sh

MODEL_ID="${MODEL_ID:-AIPlans/Qwen3-0.6b-SFT-hs2}"
DATASET_ID="${DATASET_ID:-Jennny/helpsteer2-helpfulness-preference}"

SUBSET="${SUBSET:-0}"
EVAL_SUBSET="${EVAL_SUBSET:-0}"
EPOCHS="${EPOCHS:-3}"
BETA="${BETA:-0.01}"
LR="${LR:-5e-7}"

ALPHA_MIN="${ALPHA_MIN:-0.05}"
ALPHA_MAX="${ALPHA_MAX:-0.45}"
ALPHA_INIT="${ALPHA_INIT:-0.20}"
ALPHA_LR="${ALPHA_LR:-1e-3}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_LENGTH="${MAX_LENGTH:-1024}"

LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"

REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-hl-tp-qwen3-preference}"
export WANDB_PROJECT

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-src/train_variable_alpha_ipo.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/variable_alpha_ipo}"
RUN_NAME="${RUN_NAME:-qwen3_0.6b_variable_alpha_ipo_e${EPOCHS}_bs${TRAIN_BATCH_SIZE}_beta${BETA}_lr${LR}_alpha${ALPHA_INIT}}"
OUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

echo "Variable-alpha IPO experiment"
echo "  model:        ${MODEL_ID}"
echo "  dataset:      ${DATASET_ID}"
echo "  subset:       ${SUBSET} (0 = full train split)"
echo "  eval_subset:  ${EVAL_SUBSET} (0 = full validation split)"
echo "  epochs:       ${EPOCHS}"
echo "  beta:         ${BETA}"
echo "  lr:           ${LR}"
echo "  batch:        ${TRAIN_BATCH_SIZE}"
echo "  effective:    chosen/rejected concat batch $((TRAIN_BATCH_SIZE * 2))"
echo "  alpha_min:    ${ALPHA_MIN}"
echo "  alpha_max:    ${ALPHA_MAX}"
echo "  alpha_init:   ${ALPHA_INIT}"
echo "  alpha_lr:     ${ALPHA_LR}"
echo "  report_to:    ${REPORT_TO}"
echo "  wandb_project:${WANDB_PROJECT}"
echo "  output:       ${OUT_DIR}"
echo

"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --model_id "${MODEL_ID}" \
  --dataset_id "${DATASET_ID}" \
  --subset "${SUBSET}" \
  --eval_subset "${EVAL_SUBSET}" \
  --epochs "${EPOCHS}" \
  --beta "${BETA}" \
  --lr "${LR}" \
  --alpha_min "${ALPHA_MIN}" \
  --alpha_max "${ALPHA_MAX}" \
  --alpha_init "${ALPHA_INIT}" \
  --alpha_lr "${ALPHA_LR}" \
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
  --output_dir "${OUT_DIR}"

if [[ ! -d "${OUT_DIR}/final" ]]; then
  echo "Expected saved weights at ${OUT_DIR}/final, but the directory was not found." >&2
  exit 1
fi

echo
echo "Saved weights: ${OUT_DIR}/final"
echo "Saved alpha:   ${OUT_DIR}/final/alpha_config.json"
