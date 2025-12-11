# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
from typing import Union

from models.discrete_unet import DiscreteUNetModel
from models.ema import EMA
from models.unet import UNetModel

MODEL_CONFIGS = {
    "imagenet": {
        "in_channels": 1,#3
        "model_channels": 128,#192,小
        "out_channels": 1,#3
        "num_res_blocks": 2,#3小
        "attention_resolutions": [16, 32],#[2, 4, 8]
        "dropout": 0.1,
        "channel_mult": [1, 2, 2, 4],#[1, 2, 3, 4]
        "num_classes": None,#1000
        "use_checkpoint":True,#False
        "cond_dim":208,#新添加signal
        "num_heads": 4,#4
        "num_head_channels": 64,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "lol": {
        "in_channels": 6, # 3 (Noisy) + 3 (Low Light Condition) = 6
        "model_channels": 128, 
        "out_channels": 3, # 输出 RGB
        "num_res_blocks": 2,
        "attention_resolutions": [16, 32], # 根据显存调整
        "dropout": 0.1, 
        "channel_mult": [1, 2, 3, 4],
        "num_classes": None,
        "use_checkpoint": True,
        "cond_dim": 0,    
        "num_heads": 4,
        "num_head_channels": 64,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "euvp": {
        "in_channels": 6,     # 3 (Input) + 3 (Condition)
        "model_channels": 128, 
        "out_channels": 3,
        "num_res_blocks": 2,
        "attention_resolutions": [16, 32], 
        "dropout": 0.1,
        "channel_mult": [1, 2, 3, 4],
        "num_classes": None,
        "use_checkpoint": True,
        "cond_dim": 0,        # Image Condition
        "num_heads": 4,
        "num_head_channels": 64,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "imagenet_discrete": {
        "in_channels": 3,
        "model_channels": 192,
        "out_channels": 3,
        "num_res_blocks": 4,
        "attention_resolutions": [2, 4, 8],
        "dropout": 0.2,
        "channel_mult": [2, 3, 4, 4],
        "num_classes": 1000,
        "use_checkpoint": False,
        "num_heads": -1,
        "num_head_channels": 64,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "cifar10": {
        "in_channels": 3,
        "model_channels": 128,
        "out_channels": 3,
        "num_res_blocks": 4,
        "attention_resolutions": [2],
        "dropout": 0.3,
        "channel_mult": [2, 2, 2],
        "conv_resample": False,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "cifar10_discrete": {
        "in_channels": 3,
        "model_channels": 96,
        "out_channels": 3,
        "num_res_blocks": 5,
        "attention_resolutions": [2],
        "dropout": 0.4,
        "channel_mult": [3, 4, 4],
        "conv_resample": False,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": -1,
        "num_head_channels": 64,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
}


def instantiate_model(
    architechture: str, is_discrete: bool, use_ema: bool
) -> Union[UNetModel, DiscreteUNetModel]:
    assert (
        architechture in MODEL_CONFIGS
    ), f"Model architecture {architechture} is missing its config."

    if is_discrete:
        if architechture + "_discrete" in MODEL_CONFIGS:
            config = MODEL_CONFIGS[architechture + "_discrete"]
        else:
            config = MODEL_CONFIGS[architechture]
        model = DiscreteUNetModel(
            vocab_size=257,
            **config,
        )
    else:
        print(f"[Debug] Instantiating UNet for dataset: {architechture}")
        model = UNetModel(
            **MODEL_CONFIGS[architechture],
            dataset=architechture
        )

    if use_ema:
        return EMA(model=model)
    else:
        return model
