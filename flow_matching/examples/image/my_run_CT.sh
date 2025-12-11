#!/bin/bash

export PYTHONPATH="/mnt/shared-storage-user/yangmingyuan/flow_matching:$PYTHONPATH"
export TORCH_HOME="/data/yangmingyuan/pretrained_models"

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
    /mnt/shared-storage-user/yangmingyuan/flow_matching/examples/image/train.py \
    --dataset imagenet \
    --data_path "/mnt/shared-storage-user/yangmingyuan/train.h5" \
    --output_dir "/data/yangmingyuan/output_dir/experiment_CT_001" \
    --batch_size 64
    #--resume "/data/yangmingyuan/output_dir/experiment_CT_001/checkpoint-49.pth"


#torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
#    /mnt/shared-storage-user/yangmingyuan/flow_matching/examples/image/train.py \
#    --dataset lol \
#    --data_path "/mnt/shared-storage-user/yangmingyuan/data/LOL" \
#    --output_dir "/mnt/shared-storage-user/yangmingyuan/output_dir/experiment_lol_003" \
    #--resume "/mnt/shared-storage-user/yangmingyuan/output_dir/experiment_lol_002/checkpoint.#pth"

#torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
#    /mnt/shared-storage-user/yangmingyuan/flow_matching/examples/image/train.py \
#    --dataset euvp \
#    --data_path "/mnt/shared-storage-user/yangmingyuan/data/EUVP" \
#    --output_dir "/mnt/shared-storage-user/yangmingyuan/output_dir/experiment_euvp_001" \
#    --resume "/mnt/shared-storage-user/yangmingyuan/output_dir/experiment_euvp_001/#checkpoint-49.pth"