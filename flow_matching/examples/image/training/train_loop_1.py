# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import gc
import logging
import math
from typing import Iterable
import os

import torch
import matplotlib.pyplot as plt
import numpy as np
from flow_matching.path import CondOTProbPath, MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from models.ema import EMA
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric
from training.grad_scaler import NativeScalerWithGradNormCount

logger = logging.getLogger(__name__)

MASK_TOKEN = 256
PRINT_FREQUENCY = 50


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: torch.torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScalerWithGradNormCount,
    args: argparse.Namespace,
):
    gc.collect()
    model.train(True)
    batch_loss = MeanMetric().to(device, non_blocking=True)
    epoch_loss = MeanMetric().to(device, non_blocking=True)

    accum_iter = args.accum_iter
    if args.discrete_flow_matching:
        scheduler = PolynomialConvexScheduler(n=3.0)
        path = MixtureDiscreteProbPath(scheduler=scheduler)
    else:
        path = CondOTProbPath()


    # **新增：用于记录训练过程中真实的 (t, SNR) 数据**
    log_t_steps = []
    log_snr_values = []

    for data_iter_step, batch in enumerate(data_loader):
        if data_iter_step % accum_iter == 0:
            optimizer.zero_grad()
            batch_loss.reset()
            if data_iter_step > 0 and args.test_run:
                break

        #samples对应batch['data'], lables对应batch['condition']
        samples = batch['data'].to(device, non_blocking=True)
        condition_signal = batch['condition'].to(device, non_blocking=True)


        # === [新增测试代码] 打印真实数据的形状 ===
        # 只在第一个 step 打印，防止刷屏
        if data_iter_step == 0:
            print(f"\n[DEBUG] 真实数据 samples.shape: {samples.shape}")
            print(f"[DEBUG] 当前实际 Batch Size (第一维): {samples.shape[0]}")
            # 如果使用了 DDP，这里打印的只是单张卡的 Batch Size
        # =======================================




        if torch.rand(1) < args.class_drop_prob:
            conditioning = {}
        else:
            conditioning = {"label": condition_signal}

        if args.discrete_flow_matching:
            samples = (samples * 255.0).to(torch.long)
            t = torch.torch.rand(samples.shape[0]).to(device)

            # sample probability path
            x_0 = (
                torch.zeros(samples.shape, dtype=torch.long, device=device) + MASK_TOKEN
            )
            path_sample = path.sample(t=t, x_0=x_0, x_1=samples)

            # discrete flow matching loss
            logits = model(path_sample.x_t, t=t, extra=conditioning)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape([-1, 257]), samples.reshape([-1])
            ).mean()
        else:
            # Scaling to [-1, 1] from [0, 1]
            samples = samples * 2.0 - 1.0
            noise = torch.randn_like(samples).to(device)
            if args.skewed_timesteps:
                t = skewed_timestep_sample(samples.shape[0], device=device)
            else:
                t = torch.torch.rand(samples.shape[0]).to(device)

            #######################################################
            # path.sample: x_0 is Noise, x_1 is Data (Samples)
            # x_t = (1-t)*noise + t*samples

            path_sample = path.sample(t=t, x_0=noise, x_1=samples)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t

            # **新增：计算 Training Batch 的真实 SNR**
            # SNR 定义为: Signal_Power / Noise_Power
            # 在 OT path 中，信号分量是 t * samples，噪声分量是 (1-t) * noise
            # 为了防止除以0，分母加一个极小值

            with torch.no_grad():
                # 计算能量 (Batch Mean)
                # 注意：t 的形状是 [B], 需要 reshape 才能广播
                t_view = t.view(-1, 1, 1, 1) # 假设是图像数据 B,C,H,W
                
                signal_component = t_view * samples
                noise_component = (1 - t_view) * noise
                
                # 计算功率 (Mean Squared)
                sig_p = torch.mean(signal_component**2, dim=[1, 2, 3])
                noise_p = torch.mean(noise_component**2, dim=[1, 2, 3])
                
                # 计算 dB，防止 log(0)
                snr_db = 10 * torch.log10(sig_p / (noise_p + 1e-8))
                
                # 随机采样一部分数据存下来绘图（避免存太多内存爆炸）
                # 这里每步存取 Batch 中前 10 个样本的 t 和 SNR
                log_t_steps.append(t[:10].detach().cpu().numpy())
                log_snr_values.append(snr_db[:10].detach().cpu().numpy())



            with torch.cuda.amp.autocast():
                loss = torch.pow(model(x_t, t, extra=conditioning) - u_t, 2).mean()

        loss_value = loss.item()
        batch_loss.update(loss)
        epoch_loss.update(loss)

        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter

        # Loss scaler applies the optimizer when update_grad is set to true.
        # Otherwise just updates the internal gradient scales
        apply_update = (data_iter_step + 1) % accum_iter == 0
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=apply_update,
        )
        if apply_update and isinstance(model, EMA):
            model.update_ema()
        elif (
            apply_update
            and isinstance(model, DistributedDataParallel)
            and isinstance(model.module, EMA)
        ):
            model.module.update_ema()

        lr = optimizer.param_groups[0]["lr"]
        if data_iter_step % PRINT_FREQUENCY == 0:
            logger.info(
                f"Epoch {epoch} [{data_iter_step}/{len(data_loader)}]: loss = {batch_loss.compute()}, lr = {lr}"
            )

    lr_schedule.step()



    # **新增：在 Epoch 结束时绘制 SNR 图像**
    # 只有主进程画图
    if len(log_t_steps) > 0 and args.output_dir:
        try:
            all_t = np.concatenate(log_t_steps)
            all_snr = np.concatenate(log_snr_values)
            
            # 按时间排序以便画线（如果是散点图可以不排，但折线图需要）
            sorted_indices = np.argsort(all_t)
            all_t = all_t[sorted_indices]
            all_snr = all_snr[sorted_indices]

            plt.figure(figsize=(10, 6))
            # 使用散点图因为 t 是随机采样的，折线可能会很乱
            plt.scatter(all_t, all_snr, alpha=0.5, s=1, label='Actual Training SNR')
            
            # 计算滑动平均以画出趋势线
            window_size = 100
            if len(all_snr) > window_size:
                 df_snr = np.convolve(all_snr, np.ones(window_size)/window_size, mode='valid')
                 df_t = all_t[window_size//2 : -window_size//2 + 1]
                 plt.plot(df_t, df_snr, color='red', label='Trend')

            plt.xlabel("Time Step t (0=Noise, 1=Data)")
            plt.ylabel("SNR (dB)")
            plt.title(f"Training SNR Evolution (Epoch {epoch})")
            plt.grid(True)
            plt.legend()
            
            #save_path = os.path.join(args.output_dir, f"snr_training_epoch_{epoch}.png")
            #plt.savefig(save_path)
            sub_dir = os.path.join(args.output_dir, "vis_snr_train")
            os.makedirs(sub_dir, exist_ok=True)
            save_path = os.path.join(sub_dir, f"snr_training_epoch_{epoch}.png")
            plt.savefig(save_path)



            plt.close()
            logger.info(f"Saved Training SNR plot to {save_path}")
        except Exception as e:
            logger.warning(f"Failed to plot training SNR: {e}")


    return {"loss": float(epoch_loss.compute().detach().cpu())}
