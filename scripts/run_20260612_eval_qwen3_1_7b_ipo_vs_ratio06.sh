#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export HF_HOME="${HF_HOME:-${ROOT_DIR}/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
LM_EVAL_BIN="${LM_EVAL_BIN:-${ROOT_DIR}/.venv/bin/lm_eval}"

TASKS="${TASKS:-arc_challenge,arc_easy,hellaswag,truthfulqa_mc2,winogrande}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/eval_qwen3_1_7b_ipo_vs_ratio06}"

STANDARD_MODEL="${STANDARD_MODEL:-${ROOT_DIR}/outputs/qwen3_1_7b_ipo_vs_ratio06/20260610/objective_standard_ipo_norm_none_e3_bs1_beta0.01_lr5e-7/final}"
RATIO_MODEL="${RATIO_MODEL:-${ROOT_DIR}/outputs/qwen3_1_7b_ipo_vs_ratio06/20260610/objective_target_ratio_norm_none_ratio_0.6_e3_bs1_beta0.01_lr5e-7/final}"

run_eval() {
  local name="$1"
  local model_path="$2"
  local out_dir="${OUTPUT_ROOT}/${name}"

  if [[ -f "${out_dir}/results.json" ]]; then
    echo "Reusing existing result: ${out_dir}/results.json"
    return
  fi

  mkdir -p "${out_dir}"
  echo "============================================================"
  echo "Evaluating ${name}"
  echo "Model: ${model_path}"
  echo "Tasks: ${TASKS}"
  echo "Output: ${out_dir}"
  echo "============================================================"

  "${LM_EVAL_BIN}" run \
    --model hf \
    --model_args "pretrained=${model_path},trust_remote_code=True,dtype=bfloat16" \
    --tasks "${TASKS}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${out_dir}"
}

run_eval "standard_IPO" "${STANDARD_MODEL}"
run_eval "ratioeq0p6" "${RATIO_MODEL}"

"${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path

import pandas as pd

root = Path("outputs/eval_qwen3_1_7b_ipo_vs_ratio06")
models = {
    "Qwen3-1.7B standard IPO": root / "standard_IPO" / "results.json",
    "Qwen3-1.7B ratio=0.6": root / "ratioeq0p6" / "results.json",
}
task_names = {
    "arc_challenge": "ARC-Challenge",
    "arc_easy": "ARC-Easy",
    "hellaswag": "HellaSwag",
    "truthfulqa_mc2": "TruthfulQA MC2",
    "winogrande": "Winogrande",
}

rows = []
for model, path in models.items():
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    for task, metrics in data.get("results", {}).items():
        if "acc_norm,none" in metrics:
            metric_key = "acc_norm,none"
            metric = "acc_norm"
        else:
            metric_key = "acc,none"
            metric = "acc"
        rows.append({
            "Task": task_names.get(task, task),
            "Metric": metric,
            "model": model,
            "Score": metrics.get(metric_key),
        })

if rows:
    benchmark = pd.DataFrame(rows)
    comparison = benchmark.pivot_table(index=["Task", "Metric"], columns="model", values="Score").reset_index()
    if {"Qwen3-1.7B standard IPO", "Qwen3-1.7B ratio=0.6"}.issubset(comparison.columns):
        comparison["ratio=0.6 - standard IPO"] = (
            comparison["Qwen3-1.7B ratio=0.6"] - comparison["Qwen3-1.7B standard IPO"]
        )
    comparison.to_csv(root / "benchmark_comparison.csv", index=False)
    ranking = comparison[[c for c in comparison.columns if c.startswith("Qwen3-1.7B")]].mean().rename("mean_score")
    ranking.sort_values(ascending=False).to_csv(root / "benchmark_ranking.csv")
    print(comparison)
    print()
    print(ranking.sort_values(ascending=False))
else:
    print("No benchmark results found.")
PY

echo "Benchmark evaluation finished."
