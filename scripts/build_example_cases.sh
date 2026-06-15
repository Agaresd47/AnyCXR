#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_ROOT}/weights/anychest_inference_bundle.pt}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "${REPO_ROOT}/examples/outputs"

"${PYTHON_BIN}" "${REPO_ROOT}/seg.py" \
  --input-path "${REPO_ROOT}/examples/inputs/la" \
  --output-dir "${REPO_ROOT}/examples/outputs/la" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --profile la \
  --device "${DEVICE}" \
  --batch-size 1

"${PYTHON_BIN}" "${REPO_ROOT}/seg.py" \
  --input-path "${REPO_ROOT}/examples/inputs/pa" \
  --output-dir "${REPO_ROOT}/examples/outputs/pa" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --profile pa \
  --device "${DEVICE}" \
  --batch-size 1

"${PYTHON_BIN}" "${REPO_ROOT}/seg.py" \
  --input-path "${REPO_ROOT}/examples/inputs/oblique_45" \
  --output-dir "${REPO_ROOT}/examples/outputs/oblique_45" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --profile oblique_45 \
  --device "${DEVICE}" \
  --batch-size 1

echo "Example outputs refreshed under ${REPO_ROOT}/examples/outputs"
