#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/home/bramiozo/DEV/CardioNER.nl"
MODEL_ROOT="${BASE_DIR}/output/EuroBERT-610m"
DATA_ROOT="${BASE_DIR}/assets/MultiClinNER"
INFERENCE_PIPE="dt4h"
BATCH_SIZE=4

entities="disease procedure symptom"
languages="cz sv en es it nl ro"

for entity in ${entities}; do
  ENTITY_UPPER="$(printf "%s" "${entity}" | tr '[:lower:]' '[:upper:]')"
  model_path="${MODEL_ROOT}/${ENTITY_UPPER}_3l_multilabel_weighted/fold_0"

  for lang in ${languages}; do
    corpus_inference="${DATA_ROOT}/MultiClinNER-${lang}/test/${entity}/txt"
    output_prefix="${lang}_${entity}_eurobert610_ml_silver_"

    poetry run python -m cardioner.main \
      --inference_only \
      --model_path="${model_path}" \
      --corpus_inference="${corpus_inference}" \
      --lang=multi \
      --output_file_prefix="${output_prefix}" \
      --inference_pipe="${INFERENCE_PIPE}" \
      --inference_batch_size="${BATCH_SIZE}" \
      --trust_remote_code
  done
done
