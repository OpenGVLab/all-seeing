#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=$1
DATA_DIR="playground"

# SLURM
GPUS=1
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
QUOTA_TYPE="reserved"

PROJECT_NAME="eval/${CKPT}/seed"
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
        --question-file ${DATA_DIR}/data/eval/seed_bench/llava-seed-bench.jsonl \
        --image-folder ${DATA_DIR}/data/eval/seed_bench \
        --answers-file ${DATA_DIR}/data/eval/seed_bench/answers/${CKPT}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=${DATA_DIR}/data/eval/seed_bench/answers/${CKPT}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${DATA_DIR}/data/eval/seed_bench/answers/${CKPT}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

mkdir -p ${DATA_DIR}/data/eval/seed_bench/answers_upload/${CKPT}

# Evaluate
python scripts/convert_seed_for_submission.py \
    --annotation-file ${DATA_DIR}/data/eval/seed_bench/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file ${DATA_DIR}/data/eval/seed_bench/answers_upload/${CKPT}/asmv2-13b.jsonl \
1>"logs/${PROJECT_NAME}.out" \
