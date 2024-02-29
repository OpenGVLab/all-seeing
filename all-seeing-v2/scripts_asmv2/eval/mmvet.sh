#!/bin/bash

MODEL_PATH=$1
DATA_DIR=playground

# SLURM
GPUS=1
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
QUOTA_TYPE="reserved"

PROJECT_NAME="eval/${MODEL_PATH}/mmvet"
mkdir -p "logs/$(dirname "${PROJECT_NAME}")"

srun --gres=gpu:"${GPUS_PER_NODE}" \
    --ntasks="${GPUS}" \
    --ntasks-per-node="${GPUS_PER_NODE}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --quotatype="${QUOTA_TYPE}" \
    -p INTERN2 \
python -m llava.eval.model_vqa \
    --model-path ${MODEL_PATH} \
    --question-file ${DATA_DIR}/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ${DATA_DIR}/data/eval/mm-vet/images \
    --answers-file ${DATA_DIR}/data/eval/mm-vet/answers/${MODEL_PATH}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ${DATA_DIR}/data/eval/mm-vet/results/"$(dirname "${MODEL_PATH}")"

python scripts/convert_mmvet_for_eval.py \
    --src ${DATA_DIR}/data/eval/mm-vet/answers/${MODEL_PATH}.jsonl \
    --dst ${DATA_DIR}/data/eval/mm-vet/results/${MODEL_PATH}.json

