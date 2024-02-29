#!/usr/bin/env bash

set -x

CKPT=$1
DATASET=$2

# SLURM
PROJECT_NAME="${DATASET}_${CKPT}"
mkdir -p "logs/$(dirname "${PROJECT_NAME}")"

PARTITION=INTERN2
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
QUOTA_TYPE="auto"

srun \
    -p "${PARTITION}" \
    --job-name="eval" \
    --gres=gpu:"${GPUS_PER_NODE}" \
    --ntasks="${GPUS}" \
    --ntasks-per-node="${GPUS_PER_NODE}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --quotatype="${QUOTA_TYPE}" \
    --async \
    --output="logs/${PROJECT_NAME}.out" \
    --error="logs/${PROJECT_NAME}.err" \
python -u eval_region_recognition.py \
    --checkpoint "${CKPT}" \
    --dataset "${DATASET}" \
    --batch-size 8 \
