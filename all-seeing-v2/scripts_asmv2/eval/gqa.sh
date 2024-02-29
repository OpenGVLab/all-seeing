#!/bin/bash

BASE_DIR="$(pwd)"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=$1
SPLIT="llava_gqa_testdev_balanced"
DATA_DIR="playground"
GQADIR="playground/data/eval/gqa/data"

# SLURM
GPUS=1
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
QUOTA_TYPE="reserved"

PROJECT_NAME="eval/${CKPT}/gqa"
mkdir -p "logs/$(dirname "${PROJECT_NAME}")"

for IDX in $(seq 0 $((CHUNKS-1))); do
    srun --gres=gpu:"${GPUS_PER_NODE}" \
        --ntasks="${GPUS}" \
        --ntasks-per-node="${GPUS_PER_NODE}" \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --quotatype="${QUOTA_TYPE}" \
        -p INTERN2 \
    python -m llava.eval.model_vqa_loader \
        --model-path ${CKPT} \
        --question-file ${DATA_DIR}/data/eval/gqa/$SPLIT.jsonl \
        --image-folder ${DATA_DIR}/data/eval/gqa/data/images \
        --answers-file ${DATA_DIR}/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=${DATA_DIR}/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${DATA_DIR}/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced 1>"${BASE_DIR}/logs/${PROJECT_NAME}.out"
