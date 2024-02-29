#!/bin/bash

SPLIT="mmbench_dev_cn_20231003"
MODEL_PATH=$1
DATA_DIR=playground

# SLURM
GPUS=1
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
QUOTA_TYPE="reserved"

srun --gres=gpu:"${GPUS_PER_NODE}" \
    --ntasks="${GPUS}" \
    --ntasks-per-node="${GPUS_PER_NODE}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --quotatype="${QUOTA_TYPE}" \
    -p INTERN2 \
python -m llava.eval.model_vqa_mmbench \
    --model-path ${MODEL_PATH} \
    --question-file ${DATA_DIR}/data/eval/mmbench_cn/$SPLIT.tsv \
    --answers-file ${DATA_DIR}/data/eval/mmbench_cn/answers/$SPLIT/${MODEL_PATH}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ${DATA_DIR}/data/eval/mmbench_cn/answers_upload/$SPLIT/"$(dirname "${MODEL_PATH}")"

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ${DATA_DIR}/data/eval/mmbench_cn/$SPLIT.tsv \
    --result-dir ${DATA_DIR}/data/eval/mmbench_cn/answers/$SPLIT \
    --upload-dir ${DATA_DIR}/data/eval/mmbench_cn/answers_upload/$SPLIT \
    --experiment ${MODEL_PATH}
