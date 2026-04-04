#!/bin/bash
# QLoRA + DDP across 3 GPUs.

export CUDA_VISIBLE_DEVICES=0,1,2   # adjust to your GPU IDs

# Show actual NCCL errors instead of silent hangs
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=WARN

torchrun --nproc_per_node=3 \
    --master_port=29500 \
    deepseek_ft.py 2>&1 | tee train.log
