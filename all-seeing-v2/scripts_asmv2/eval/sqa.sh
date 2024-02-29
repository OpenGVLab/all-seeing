#!/bin/bash

MODEL_PATH=$1
DATA_DIR=playground

# SLURM
GPUS=1
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
QUOTA_TYPE="reserved"

PROJECT_NAME="eval/${MODEL_PATH}/sqa"
mkdir -p "logs/$(dirname "${PROJECT_NAME}")"

srun --gres=gpu:"${GPUS_PER_NODE}" \
    --ntasks="${GPUS}" \
    --ntasks-per-node="${GPUS_PER_NODE}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --quotatype="${QUOTA_TYPE}" \
    -p INTERN2 \
python -m llava.eval.model_vqa_science \
    --model-path ${MODEL_PATH} \
    --question-file ${DATA_DIR}/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ${DATA_DIR}/data/eval/scienceqa/images/test \
    --answers-file ${DATA_DIR}/data/eval/scienceqa/answers/${MODEL_PATH}/asmv2-13b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ${DATA_DIR}/data/eval/scienceqa \
    --result-file ${DATA_DIR}/data/eval/scienceqa/answers/${MODEL_PATH}/asmv2-13b.jsonl \
    --output-file ${DATA_DIR}/data/eval/scienceqa/answers/${MODEL_PATH}/asmv2-13b_output.jsonl \
    --output-result ${DATA_DIR}/data/eval/scienceqa/answers/${MODEL_PATH}/asmv2-13b_result.json \
1>"logs/${PROJECT_NAME}.out" \
