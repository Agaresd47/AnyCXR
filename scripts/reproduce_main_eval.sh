#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${DATASET_DIR:-/home/bruce/data/cst/Dataset003_Full/test_case}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_ROOT}/weights/anychest_inference_bundle.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/repro_outputs}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-4}"

mkdir -p "${OUTPUT_ROOT}"

run_eval() {
  local profile="$1"
  local output_dir="${OUTPUT_ROOT}/${profile}"
  echo "==> Reproducing ${profile}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_test_case.py" \
    --dataset-dir "${DATASET_DIR}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --profile "${profile}" \
    --output-dir "${output_dir}" \
    --device "${DEVICE}" \
    --batch-size "${BATCH_SIZE}"
}

run_eval "la"
run_eval "pa"
run_eval "oblique"

echo "Finished. Reports written under ${OUTPUT_ROOT}."
