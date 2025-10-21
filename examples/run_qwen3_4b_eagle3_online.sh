#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# support tp8 train eagle3 for Qwen3-4B/8B/32B up to tp_size = 8
NUM_GPUS=4

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online.py \
    --target-model-path /home/zhaode.wzd/workspace/Qwen3-4B-Instruct-2507 \
    --draft-model-config $ROOT_DIR/configs/qwen3-4b-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --output-dir $ROOT_DIR/outputs/Qwen3-4B-eagle3 \
    --build-dataset-num-proc 64 \
    --num-epochs 10 \
    --batch-size 16 \
    --draft-micro-batch-size 16 \
    --draft-global-batch-size 64 \
    --tp-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --ttt-length 7 \
    --report-to swanlab \
    --swanlab-name Qwen3-4B-EAGLE3