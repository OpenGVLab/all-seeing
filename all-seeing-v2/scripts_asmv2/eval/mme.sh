#!/bin/bash

MODEL_PATH=$1
DATA_DIR=playground
BASE_DIR="$(pwd)"

# SLURM
GPUS=1
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
QUOTA_TYPE="reserved"

PROJECT_NAME="eval/${MODEL_PATH}/mme"
mkdir -p "logs/$(dirname "${PROJECT_NAME}")"

srun --gres=gpu:"${GPUS_PER_NODE}" \
    --ntasks="${GPUS}" \
    --ntasks-per-node="${GPUS_PER_NODE}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --quotatype="${QUOTA_TYPE}" \
    -p INTERN2 \
python -m llava.eval.model_vqa_loader \
    --model-path ${MODEL_PATH} \
    --question-file ${DATA_DIR}/data/eval/MME/llava_mme.jsonl \
    --image-folder ${DATA_DIR}/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ${DATA_DIR}/data/eval/MME/answers/${MODEL_PATH}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ${DATA_DIR}/data/eval/MME

python convert_answer_to_mme.py --experiment ${MODEL_PATH}

cd eval_tool

python calculation.py --results_dir answers/${MODEL_PATH} 1>"${BASE_DIR}/logs/${PROJECT_NAME}.out"
