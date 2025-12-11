#!/bin/bash

export PYTHONPATH="/mnt/shared-storage-user/yangmingyuan/flow_matching:$PYTHONPATH"


python /mnt/shared-storage-user/yangmingyuan/flow_matching/examples/image/train.py \
    --dataset imagenet \
    --data_path "/mnt/shared-storage-user/yangmingyuan/train.h5" \
    --output_dir "/mnt/shared-storage-user/yangmingyuan/output_dir/experiment_002" \