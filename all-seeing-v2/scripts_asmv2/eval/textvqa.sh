#!/bin/bash

MODEL_PATH=$1
DATA_DIR=playground

# SLURM
GPUS=1
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
QUOTA_TYPE="reserved"

PROJECT_NAME="eval/${MODEL_PATH}/textvqa"
mkdir -p "logs/$(dirname "${PROJECT_NAME}")"

srun --gres=gpu:"${GPUS_PER_NODE}" \
    --ntasks="${GPUS}" \
    --ntasks-per-node="${GPUS_PER_NODE}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --quotatype="${QUOTA_TYPE}" \
    -p INTERN2 \
python -m llava.eval.model_vqa_loader \
    --model-path ${MODEL_PATH} \
    --question-file ${DATA_DIR}/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ${DATA_DIR}/data/eval/textvqa/train_images \
    --answers-file ${DATA_DIR}/data/eval/textvqa/answers/${MODEL_PATH}/asmv2-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ${DATA_DIR}/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ${DATA_DIR}/data/eval/textvqa/answers/${MODEL_PATH}/asmv2-13b.jsonl \
1>"logs/${PROJECT_NAME}.out"
