#!/bin/bash

export PYTHONPATH="/mnt/shared-storage-user/yangmingyuan/flow_matching:$PYTHONPATH"
export TORCH_HOME="/data/yangmingyuan/pretrained_models"

torchrun --nproc_per_node=3 --nnodes=1 --node_rank=0 \
    /mnt/shared-storage-user/yangmingyuan/flow_matching/examples/image/train.py \
    --dataset imagenet \
    --data_path "/mnt/shared-storage-user/yangmingyuan/val.h5" \
    --output_dir "/data/yangmingyuan/eval_results/eval_results_ct_002" \
    --resume "/data/yangmingyuan/output_dir/experiment_CT_001/checkpoint-499.pth" \
    --eval_only \
    --compute_fid \
    --fid_samples 300 \



#torchrun --nproc_per_node=3 --nnodes=1 --node_rank=0 \
#    /mnt/shared-storage-user/yangmingyuan/flow_matching/examples/image/train.py \
#    --dataset lol \
#    --data_path "/data/yangmingyuan/data/LOL" \
#    --output_dir "/data/yangmingyuan/eval_results/eval_results_lol_003" \
#    --resume "/data/yangmingyuan/output_dir/experiment_lol_001/checkpoint-249.pth" \
#    --eval_only \
#    --compute_fid \
#    --fid_samples 300 \
#    --cfg_scale 0.0