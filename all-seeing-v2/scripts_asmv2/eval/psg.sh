#!/bin/bash

MODEL_PATH=$1
DATA_DIR=playground

# SLURM
GPUS=1
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
QUOTA_TYPE="reserved"

PROJECT_NAME="eval/${MODEL_PATH}/psg"
mkdir -p "logs/$(dirname "${PROJECT_NAME}")"

srun --gres=gpu:"${GPUS_PER_NODE}" \
    --ntasks="${GPUS}" \
    --ntasks-per-node="${GPUS_PER_NODE}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --quotatype="${QUOTA_TYPE}" \
    -p INTERN2 \
python -m llava.eval.model_vqa_loader \
    --model-path ${MODEL_PATH} \
    --question-file ${DATA_DIR}/data/eval/psg/psg_eval.jsonl \
    --image-folder ${DATA_DIR}/data \
    --answers-file ${DATA_DIR}/data/eval/psg/answers/${MODEL_PATH}/asmv2-13b.jsonl \
    --temperature 0 \
    --max_new_tokens 1536 \
    --conv-mode vicuna_v1

python llava/eval/eval_psg.py \
    --question-file ${DATA_DIR}/data/eval/psg/psg_eval.jsonl \
    --result-file ${DATA_DIR}/data/eval/psg/answers/${MODEL_PATH}/asmv2-13b.jsonl \
    --square-pad-postprocess \
1>"logs/${PROJECT_NAME}.out" \
