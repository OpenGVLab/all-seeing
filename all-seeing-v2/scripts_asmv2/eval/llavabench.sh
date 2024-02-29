#!/bin/bash

MODEL_PATH=$1
DATA_DIR=playground

# SLURM
GPUS=1
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
QUOTA_TYPE="reserved"

PROJECT_NAME="eval/${MODEL_PATH}/llavabench"
mkdir -p "logs/$(dirname "${PROJECT_NAME}")"

srun --gres=gpu:"${GPUS_PER_NODE}" \
    --ntasks="${GPUS}" \
    --ntasks-per-node="${GPUS_PER_NODE}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --quotatype="${QUOTA_TYPE}" \
    -p INTERN2 \
python -m llava.eval.model_vqa \
    --model-path ${MODEL_PATH} \
    --question-file ${DATA_DIR}/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ${DATA_DIR}/data/eval/llava-bench-in-the-wild/images \
    --answers-file ${DATA_DIR}/data/eval/llava-bench-in-the-wild/answers/${MODEL_PATH}/asmv2-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ${DATA_DIR}/data/eval/llava-bench-in-the-wild/reviews/${MODEL_PATH}

# 379
export OPENAI_API_KEY="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"

python -u llava/eval/eval_gpt_review_bench.py \
    --question ${DATA_DIR}/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context ${DATA_DIR}/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        ${DATA_DIR}/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        ${DATA_DIR}/data/eval/llava-bench-in-the-wild/answers/${MODEL_PATH}/asmv2-13b.jsonl \
    --output \
        ${DATA_DIR}/data/eval/llava-bench-in-the-wild/reviews/${MODEL_PATH}/asmv2-13b.jsonl

python llava/eval/summarize_gpt_review.py -f ${DATA_DIR}/data/eval/llava-bench-in-the-wild/reviews/${MODEL_PATH}/asmv2-13b.jsonl \
1>"logs/${PROJECT_NAME}.out" \
