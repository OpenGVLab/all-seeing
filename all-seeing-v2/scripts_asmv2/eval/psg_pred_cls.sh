#!/bin/bash

MODEL_PATH=$1
DATA_DIR=playground

# SLURM
GPUS=8
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
QUOTA_TYPE="reserved"

PROJECT_NAME="eval/${MODEL_PATH}/psg_pred_cls_full"
mkdir -p "logs/$(dirname "${PROJECT_NAME}")"

output_file=${DATA_DIR}/data/eval/psg_pred_cls/answers_full/${MODEL_PATH}/asmv2-13b.jsonl

srun --gres=gpu:"${GPUS_PER_NODE}" \
    --ntasks="${GPUS}" \
    --ntasks-per-node="${GPUS_PER_NODE}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --quotatype="${QUOTA_TYPE}" \
    -p INTERN2 \
python -m llava.eval.model_vqa_loader_vocab_rank \
    --model-path ${MODEL_PATH} \
    --options-file ${DATA_DIR}/data/eval/psg_pred_cls/psg_pred_cls_options.json \
    --question-file ${DATA_DIR}/data/eval/psg_pred_cls/psg_pred_cls_full_square_pad.jsonl \
    --image-folder ${DATA_DIR}/data \
    --answers-file ${output_file} \
    --conv-mode vicuna_v1

# Clear out the output file if it exists.
> "$output_file"
# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((GPUS-1))); do
    cat ${DATA_DIR}/data/eval/psg_pred_cls/answers_full/${MODEL_PATH}/asmv2-13b_${IDX}.jsonl >> "$output_file"
done

python llava/eval/eval_psg_pred_cls.py \
    --question-file ${DATA_DIR}/data/eval/psg_pred_cls/psg_pred_cls_full_square_pad.jsonl \
    --result-file ${output_file} \
1>"logs/${PROJECT_NAME}.out" \
