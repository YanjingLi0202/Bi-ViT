#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
GPUS=$3
GPUS_PER_NODE=2
PY_ARGS=${@:4}
now=$(date +"%Y%m%d_%H%M%S")

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    python -u main_2bit.py --launcher slurm ${PY_ARGS} 