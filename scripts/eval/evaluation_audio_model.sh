#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# CHUNKS=${#GPULIST[@]}
CHUNKS=1

SPLIT="llava_gqa_testdev_balanced"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
# GQADIR="/home/ai/data/llava/dataset/eval/gqa"

MODEL_PATH="/home/user27/outputs/checkpoints/llava_factory/tiny-llava-phi-2-xeus-first_try-pretrain/checkpoint-1000"
MODEL_NAME="tiny-llava-phi-2-xeus-first_try-pretrain"
EVAL_DIR="/home/ai/data/llava/dataset/eval"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m tinyllava.eval.custom_eval \
        --model-path $MODEL_PATH \
        --data_path /home/user27/LibriSpeech/train.json \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode phi &
done

wait

# output_file=$EVAL_DIR/gqa/answers/$SPLIT/$MODEL_NAME/merge.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat $EVAL_DIR/gqa/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done

# python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

# cd $GQADIR
# python eval/eval.py --tier testdev_balanced

