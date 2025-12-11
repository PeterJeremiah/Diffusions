# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.

import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets

from signal_image_dataset import SignalImageDataset

from models.model_configs import instantiate_model
from train_arg_parser import get_args_parser

from training import distributed_mode
from training.data_transform import get_train_transform
from training.eval_loop import eval_model
from training.grad_scaler import NativeScalerWithGradNormCount as NativeScaler
from training.load_and_save import load_model, save_model
from training.train_loop import train_one_epoch

logger = logging.getLogger(__name__)
@torch.no_grad()
def debug_sampling_snr_plot(model, device, args, epoch):
    """
    执行一次 Euler 采样，并记录每一步的估计 SNR。
    针对配置: 自适应 LOL (Image) 或 CT (Vector).
    """
    model.eval()
    
    # === 1. 通用参数设置 ===
    num_samples = 4        # 采样样本数
    steps = 50             # 采样步数
    dt = 1.0 / steps
    
    # === 2. 根据任务初始化 图像(x) 和 条件(conditioning) ===
    ts = []
    snrs = []
    conditioning = {}
    
    # 初始化 x 的容器，具体形状在下面定
    x = None 

    if args.dataset == "lol":
        # === LOL 任务逻辑 ===
        # 1. 确定形状 (假设 Transform Resize 到了 256x256)
        C, H, W = 3, 256, 256 
        
        # 2. 初始化噪声 x
        x = torch.randn(num_samples, C, H, W, device=device)
        
        # 3. 构造条件 (Image Condition)
        # 随机生成一个模拟的 Low Light 图像
        cond_img = torch.randn(num_samples, C, H, W, device=device)
        conditioning = {"label": cond_img}
        
    elif args.dataset == "imagenet":
        # === CT 任务逻辑 ===
        # 1. 确定形状
        C, H, W = 1, 256, 256
        cond_dim = 208 # CT 任务的 signal 维度
        
        # 2. 初始化噪声 x
        x = torch.randn(num_samples, C, H, W, device=device)
        
        # 3. 构造条件 (Vector Condition)
        random_signal = torch.randn(num_samples, cond_dim, device=device)
        conditioning = {"label": random_signal}

    elif args.dataset == "euvp": # <--- [新增] EUVP 独立分支
        # --- EUVP 任务 ---
        # EUVP 原生就是 256x256
        C, H, W = 3, 256, 256 
        
        x = torch.randn(num_samples, C, H, W, device=device)
        cond_img = torch.randn(num_samples, C, H, W, device=device)
        conditioning = {"label": cond_img}
        
    else:
        # Fallback (防止 crash)
        logger.warning(f"Debug plot not implemented for dataset {args.dataset}")
        model.train()
        return

    # === 3. 开始采样并记录 SNR ===
    logger.info(f"Running Debug Sampling (Epoch {epoch}) | Dataset: {args.dataset} | Shape: {C}x{H}x{W}")

    for i in range(steps):
        t_value = i / steps
        t = torch.full((num_samples,), t_value, device=device)
        
        # 模型预测向量场 v
        v = model(x, t, extra=conditioning)
        
        # 广播 t 以匹配 x 的形状 [B, C, H, W]
        t_view = t.view(-1, 1, 1, 1)
        
        # 核心 SNR 计算公式
        x1_pred = x + (1 - t_view) * v  # 预测的纯净信号
        x0_pred = x - t_view * v        # 预测的纯净噪声
        
        # 计算功率
        sig_p = torch.mean(x1_pred**2, dim=[1, 2, 3])
        noise_p = torch.mean(x0_pred**2, dim=[1, 2, 3])
        
        # 计算 SNR (dB)
        cur_snr = 10 * torch.log10(sig_p / (noise_p + 1e-8))
        avg_snr = cur_snr.mean().item()
        
        ts.append(t_value)
        snrs.append(avg_snr)
        
        # Euler Step update
        x = x + v * dt
        
    # === 4. 画图 ===
    if args.output_dir:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(ts, snrs, label='Estimated Sampling SNR', linewidth=2, color='blue')
            plt.xlabel("Time Step t (0=Noise -> 1=Data)")
            plt.ylabel("Estimated SNR (dB)")
            plt.title(f"Sampling SNR Evolution (Epoch {epoch})\nDataset: {args.dataset}")
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.legend()
            
            # 保存到专门的文件夹
            sub_dir = os.path.join(args.output_dir, "vis_snr_sampling")
            os.makedirs(sub_dir, exist_ok=True)
            save_path = os.path.join(sub_dir, f"snr_sampling_epoch_{epoch}.png")

            plt.savefig(save_path)
            plt.close()
            logger.info(f"Saved Sampling SNR plot to {save_path}")
        except Exception as e:
            logger.warning(f"Failed to plot sampling SNR: {e}")

    model.train() # 恢复训练模式


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    distributed_mode.init_distributed_mode(args)

    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(", ", ",\n"))
    if distributed_mode.is_main_process():
        args_filepath = Path(args.output_dir) / "args.json"
        logger.info(f"Saving args to {args_filepath}")
        with open(args_filepath, "w") as f:
            json.dump(vars(args), f)

    device = torch.device(args.device)
    print("device:",device)
    print("args.device:",args.device)

    # fix the seed for reproducibility
    seed = args.seed + distributed_mode.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    logger.info(f"Initializing Dataset: {args.dataset}")
    transform_train = get_train_transform()
    #if args.dataset == "imagenet":
    #    dataset_train = datasets.ImageFolder(args.data_path, transform=transform_train)
    if args.dataset == "imagenet": # 替换
        logger.info("使用H5 SignalImageDataset...")
        dataset_train = SignalImageDataset(
            data_path=args.data_path,
            transform=transform_train,
            mode='train',
            dataset_name="imagenet"
        )
    elif args.dataset == "lol":
        # LOL 任务 (Folder)
        logger.info("使用 LOL SignalImageDataset (Folder)...")
        dataset_train = SignalImageDataset(
            data_path=args.data_path,
            transform=transform_train, 
            mode='train',
            dataset_name="lol"
        )
    elif args.dataset == "euvp":
        # EUVP 任务 (Folder)
        logger.info("使用 EUVP SignalImageDataset (Folder)...")
        dataset_train = SignalImageDataset(
            data_path=args.data_path,
            transform=transform_train,
            mode='train',
            dataset_name="euvp"
        )
    
    elif args.dataset == "cifar10":
        dataset_train = datasets.CIFAR10(
            root=args.data_path,
            train=True,
            download=True,
            transform=transform_train,
        )
    else:
        raise NotImplementedError(f"Unsupported dataset {args.dataset}")

    logger.info(dataset_train)

    logger.info("Intializing DataLoader")
    num_tasks = distributed_mode.get_world_size()
    global_rank = distributed_mode.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    logger.info(str(sampler_train))

    # define the model
    logger.info("Initializing Model")
    model = instantiate_model(
        architechture=args.dataset,
        is_discrete=args.discrete_flow_matching,
        use_ema=args.use_ema,
    )

    model.to(device)

    model_without_ddp = model
    logger.info(str(model_without_ddp))

    eff_batch_size = (
        args.batch_size * args.accum_iter * distributed_mode.get_world_size()
    )

    logger.info(f"Learning rate: {args.lr:.2e}")

    logger.info(f"Accumulate grad iterations: {args.accum_iter}")
    logger.info(f"Effective batch size: {eff_batch_size}")

    #######################
    #原先为find_unused_parameters=True
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(
        model_without_ddp.parameters(), lr=args.lr, betas=args.optimizer_betas
    )
    if args.decay_lr:
        lr_schedule = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=args.epochs,
            start_factor=1.0,
            end_factor=1e-8 / args.lr,
        )
    else:
        lr_schedule = torch.optim.lr_scheduler.ConstantLR(
            optimizer, total_iters=args.epochs, factor=1.0
        )

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Learning-Rate Schedule: {lr_schedule}")

    loss_scaler = NativeScaler()

    load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        lr_schedule=lr_schedule,
    )

    logger.info(f"Start from {args.start_epoch} to {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if not args.eval_only:
            train_stats = train_one_epoch(
                model=model,
                data_loader=data_loader_train,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                device=device,
                epoch=epoch,
                loss_scaler=loss_scaler,
                args=args,
            )
            # ================= [INSERT THIS] =================
            # 每隔一定的 epoch 画一次采样 SNR 图
            if distributed_mode.is_main_process() and (epoch % 1 == 0):
                try:
                    debug_sampling_snr_plot(model_without_ddp, device, args, epoch)
                except Exception as e:
                    logger.warning(f"Error in debug_sampling_snr_plot: {e}")
            # =================================================
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }
        else:
            log_stats = {
                "epoch": epoch,
            }

        if args.output_dir and (
            (args.eval_frequency > 0 and (epoch + 1) % args.eval_frequency == 0)
            or args.eval_only
            or args.test_run
        ):
            if not args.eval_only:
                save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    lr_schedule=lr_schedule,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                )
            if args.distributed:
                data_loader_train.sampler.set_epoch(0)
            if distributed_mode.is_main_process():
                fid_samples = args.fid_samples - (num_tasks - 1) * (
                    args.fid_samples // num_tasks
                )
            else:
                fid_samples = args.fid_samples // num_tasks
            eval_stats = eval_model(
                model,
                data_loader_train,
                device,
                epoch=epoch,
                fid_samples=fid_samples,
                args=args,
            )
            log_stats.update({f"eval_{k}": v for k, v in eval_stats.items()})

        if args.output_dir and distributed_mode.is_main_process():
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.test_run or args.eval_only:
            break

    # =========================================================
    # 如果是纯评估模式，直接运行一次评估，不进循环
    if args.eval_only:
        logger.info("Running evaluation only...")
        if distributed_mode.is_main_process():
            fid_samples = args.fid_samples - (num_tasks - 1) * (
                args.fid_samples // num_tasks
            )
        else:
            fid_samples = args.fid_samples // num_tasks
        
        eval_stats = eval_model(
            model,
            data_loader_train, # 或者 data_loader_val
            device,
            epoch=args.start_epoch, # 使用加载时的 epoch
            fid_samples=fid_samples,
            args=args,
        )
        logger.info(f"Evaluation stats: {eval_stats}")
        return # 评估完直接结束程序
    # =========================================================

    # =========================================================
    if args.output_dir and not args.eval_only and not args.test_run:
        logger.info("End of training loop. Saving final checkpoint...")
        save_model(
            args=args,
            model=model,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            loss_scaler=loss_scaler,
            epoch=args.epochs - 1, # 标记为最后一个 epoch
        )
    # =========================================================


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

