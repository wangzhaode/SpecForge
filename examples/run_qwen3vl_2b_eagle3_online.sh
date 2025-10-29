#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

NUM_GPUS=4
MICRO_BATCH_SIZE=16
GLOBAL_BATCH_SIZE=$((MICRO_BATCH_SIZE * NUM_GPUS))

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online.py \
    --target-model-path /home/zhaode.wzd/workspace/Qwen3-VL-2B-Instruct \
    --draft-model-config $ROOT_DIR/configs/qwen3-vl-2b-eagle3.json \
    --train-data-path /home/zhaode.wzd/workspace/EagleChat/eagle_chat.jsonl \
    --output-dir $ROOT_DIR/outputs/Qwen3-VL-2B-Instruct-eagle3 \
    --build-dataset-num-proc 64 \
    --num-epochs 10 \
    --batch-size $MICRO_BATCH_SIZE \
    --draft-micro-batch-size $MICRO_BATCH_SIZE \
    --draft-global-batch-size $GLOBAL_BATCH_SIZE \
    --tp-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.language_model.embed_tokens.weight \
    --ttt-length 7 \
    --report-to swanlab \
    --swanlab-name Qwen3-VL-2B-Instruct-EAGLE3 \
    --swanlab-key 2xZYFZWHSVuTSXNoebxHc