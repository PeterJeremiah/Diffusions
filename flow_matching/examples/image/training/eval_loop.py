# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import gc
import logging
#import ospip
from argparse import Namespace
from pathlib import Path
from typing import Iterable
import json

import PIL.Image

import torch
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils import ModelWrapper
from models.discrete_unet import DiscreteUNetModel
from models.ema import EMA
from torch.nn.modules import Module
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
from torchvision.utils import save_image
from training import distributed_mode
from training.edm_time_discretization import get_time_discretization
from training.train_loop import MASK_TOKEN

logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 50


class CFGScaledModel(ModelWrapper):
    def __init__(self, model: Module):
        super().__init__(model)
        self.nfe_counter = 0

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cfg_scale: float, label: torch.Tensor
    ):

        # =========================================================
        # [DETECTIVE D] 检查喂给模型的 Label (Condition) 到底长什么样
        # =========================================================
        # 为了防止刷屏，我们只在 t 接近 0 (采样刚开始) 或者随机打印一次
        # 这里简单起见，打印 label 的统计信息
        if self.nfe_counter == 0: # 只在每张图采样的第一步打印
            print(f"\n[DETECTIVE D] Input Label to Model:")
            print(f"   -> Min: {label.min().item():.4f}")
            print(f"   -> Max: {label.max().item():.4f}")
            print(f"   -> Mean: {label.mean().item():.4f}")
            
            # 顺便检查一下输入噪声 x 的范围，确保它是标准的 Gaussian
            print(f"   -> Input Noise x: Min={x.min().item():.4f}, Max={x.max().item():.4f}")
            
            # 这是一个关键判断：
            if label.min().item() >= 0.0 and label.max().item() <= 1.0:
                 print("   [!!! WARNING !!!] Label range looks like [0, 1]. Did you train with [-1, 1]?")
            elif label.min().item() < -0.9:
                 print("   [OK] Label range looks like [-1, 1].")
        # =========================================================




        module = (
            self.model.module
            if isinstance(self.model, DistributedDataParallel)
            else self.model
        )
        is_discrete = isinstance(module, DiscreteUNetModel) or (
            isinstance(module, EMA) and isinstance(module.model, DiscreteUNetModel)
        )
        assert (
            cfg_scale == 0.0 or not is_discrete
        ), f"Cfg scaling does not work for the logit outputs of discrete models. Got cfg weight={cfg_scale} and model {type(self.model)}."
        t = torch.zeros(x.shape[0], device=x.device) + t

        if cfg_scale != 0.0:
            with torch.cuda.amp.autocast(), torch.no_grad():
                # 1. 有条件推理
                conditional = self.model(x, t, extra={"label": label})
                
                # 2. 无条件推理 (Condition Free)
                extra_uncond = {}
                
                # === [适配] LOL 任务需要全零图像作为无条件输入 ===
                # 获取底层的 UNetModel
                base_model = module
                if isinstance(base_model, EMA):
                    base_model = base_model.model
                
                # 检查 dataset 属性 (这是我们在 unet.py 里新增的)
                if hasattr(base_model, "dataset") and base_model.dataset == "lol":
                    # label 是 (B, 3, H, W) 的图像
                    zeros_img = torch.zeros_like(label)
                    extra_uncond = {"label": zeros_img}
                
                condition_free = self.model(x, t, extra=extra_uncond)

            result = (1.0 + cfg_scale) * conditional - cfg_scale * condition_free
        else:
            # Model is fully conditional, no cfg weighting needed
            with torch.cuda.amp.autocast(), torch.no_grad():
                result = self.model(x, t, extra={"label": label})

        self.nfe_counter += 1
        if is_discrete:
            return torch.softmax(result.to(dtype=torch.float32), dim=-1)
        else:
            return result.to(dtype=torch.float32)

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter


def eval_model(
    model: DistributedDataParallel,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    fid_samples: int,
    args: Namespace,
):
    gc.collect()
    cfg_scaled_model = CFGScaledModel(model=model)
    cfg_scaled_model.train(False)

    # === Initialize Solver (Discrete or Continuous) ===
    if args.discrete_flow_matching:
        scheduler = PolynomialConvexScheduler(n=3.0)
        path = MixtureDiscreteProbPath(scheduler=scheduler)
        p = torch.zeros(size=[257], dtype=torch.float32, device=device)
        p[256] = 1.0
        solver = MixtureDiscreteEulerSolver(
            model=cfg_scaled_model,
            path=path,
            vocabulary_size=257,
            source_distribution_p=p,
        )
    else:
        solver = ODESolver(velocity_model=cfg_scaled_model)
        ode_opts = args.ode_options

    # === Initialize Metrics (Generalized) ===
    logger.info("Initializing metrics: PSNR, SSIM, LPIPS...")
    # PSNR & SSIM: Can handle generic (N, C, H, W)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device, non_blocking=True)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device, non_blocking=True)
    # LPIPS: Requires 3 channels. We initialize it here, inputs will be adapted in the loop.
    # normalize=True means it expects inputs in [0,1] and will internally scale to [-1,1] for VGG.
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device, non_blocking=True)


    num_synthetic = 0
    snapshots_saved = False
    if args.output_dir:
        (Path(args.output_dir) / "snapshots").mkdir(parents=True, exist_ok=True)


    # === Evaluation Loop ===
    #for data_iter_step, (samples, labels) in enumerate(data_loader):
    for data_iter_step, batch in enumerate(data_loader):
        #samples = samples.to(device, non_blocking=True)
        #labels = labels.to(device, non_blocking=True)
        if num_synthetic >= fid_samples:
            break

        # Load Real Data
        real_samples = batch['data'].to(device, non_blocking=True)
        labels = batch['condition'].to(device, non_blocking=True)
        batch_size = real_samples.shape[0]

        # =========================================================
        # 【核心修复】将 Label (条件图) 从 [0, 1] 映射到 [-1, 1]
        # =========================================================
        if args.dataset == 'lol' or args.dataset == 'euvp':
             labels = labels * 2.0 - 1.0
        else:
             # 对于 ImageNet 等本身可能已经是 [-1, 1] 的数据，保持原样
             # 或者你可以加一个自适应判断
             labels = labels
        # =========================================================


        # =========================================================================
        # [DETECTIVE A] Check input data range from DataLoader
        # =========================================================================
        if data_iter_step == 0:
            print(f"\n[DETECTIVE A] Real Samples Raw (From DataLoader): "
                  f"Min={real_samples.min().item():.4f}, Max={real_samples.max().item():.4f}, "
                  f"Mean={real_samples.mean().item():.4f}")



        # === Sampling Process ===
        cfg_scaled_model.reset_nfe_counter()
        
        if args.discrete_flow_matching:
            x_0 = torch.zeros(real_samples.shape, dtype=torch.long, device=device) + MASK_TOKEN
            
            # Sym function logic
            if args.sym_func:
                sym = lambda t: 12.0 * torch.pow(t, 2.0) * torch.pow(1.0 - t, 0.25)
            else:
                sym = args.sym
            
            dtype = torch.float64 if args.sampling_dtype == "float64" else torch.float32

            synthetic_samples = solver.sample(
                x_init=x_0,
                step_size=1.0 / args.discrete_fm_steps,
                verbose=False,
                div_free=sym,
                dtype_categorical=dtype,
                label=labels,
                cfg_scale=args.cfg_scale,
            )
        else:
            # Continuous: Standard Gaussian Noise
            x_0 = torch.randn(real_samples.shape, dtype=torch.float32, device=device)

            if args.edm_schedule:
                time_grid = get_time_discretization(nfes=ode_opts["nfe"])
            else:
                time_grid = torch.tensor([0.0, 1.0], device=device)

            synthetic_samples = solver.sample(
                time_grid=time_grid,
                x_init=x_0,
                method=args.ode_method,
                return_intermediates=False,
                atol=ode_opts.get("atol", 1e-5),
                rtol=ode_opts.get("rtol", 1e-5),
                step_size=ode_opts.get("step_size", None),
                label=labels,
                cfg_scale=args.cfg_scale,
            )


        # =========================================================================
        # [DETECTIVE B] Check model output range
        # =========================================================================
        if data_iter_step == 0:
            print(f"[DETECTIVE B] Synthetic Raw (Model Output): "
                  f"Min={synthetic_samples.min().item():.4f}, Max={synthetic_samples.max().item():.4f}")





        # === Post-processing (Standardize to [0, 1] range) ===
        # Assume model output is [-1, 1], normalize to [0, 1]
        synthetic_samples = synthetic_samples * 0.5 + 0.5
        synthetic_samples = torch.clamp(synthetic_samples, min=0.0, max=1.0)
        
        # Simulate 8-bit quantization (Important for fair PSNR/SSIM)
        synthetic_samples = torch.floor(synthetic_samples * 255) / 255.0
        synthetic_samples = synthetic_samples.to(dtype=torch.float32)

        # Process Real samples similarly (Assuming [-1, 1] input)
        #真实图处理：去掉缩放！输入已经是 [0, 1]
        #real_samples_norm = real_samples * 0.5 + 0.5
        real_samples_norm = real_samples

        real_samples_norm = torch.clamp(real_samples_norm, min=0.0, max=1.0)
        real_samples_norm = torch.floor(real_samples_norm * 255) / 255.0
        real_samples_norm = real_samples_norm.to(dtype=torch.float32)


        # =========================================================================
        # [DETECTIVE C] Check final range going into PSNR
        # =========================================================================
        if data_iter_step == 0:
            print(f"[DETECTIVE C] Final Input to Metrics:")
            print(f"   -> Synthetic (Final): Min={synthetic_samples.min().item():.4f}, Max={synthetic_samples.max().item():.4f}")
            print(f"   -> Real (Final):      Min={real_samples_norm.min().item():.4f}, Max={real_samples_norm.max().item():.4f}\n")


        # === 4. Update Metrics (Adaptive Logic) ===
        
        # PSNR & SSIM:
        # Use original channels (Whether 1 or 3).
        # Ensures metrics reflect the actual generation task (grayscale vs color).
        psnr_metric.update(synthetic_samples, real_samples_norm)
        ssim_metric.update(synthetic_samples, real_samples_norm)

        # LPIPS:
        # Requires strictly 3 channels (RGB).
        # Adaptation Logic: 
        # - If 1 channel (N,1,H,W) -> Repeat to (N,3,H,W)
        # - If 3 channels -> Pass as is
        # - If >3 channels -> Take first 3 (Rare case, but safe)
        if synthetic_samples.shape[1] == 1:
            syn_for_lpips = synthetic_samples.repeat(1, 3, 1, 1)
            real_for_lpips = real_samples_norm.repeat(1, 3, 1, 1)
        elif synthetic_samples.shape[1] >= 3:
            syn_for_lpips = synthetic_samples[:, :3, :, :]
            real_for_lpips = real_samples_norm[:, :3, :, :]
        else:
            # Fallback for 2 channels? Usually repeat logic or pad.
            # Assuming standard image data, repeat is safest generic fallback.
            syn_for_lpips = synthetic_samples.repeat(1, 3, 1, 1)[:, :3, :, :]
            real_for_lpips = real_samples_norm.repeat(1, 3, 1, 1)[:, :3, :, :]

        lpips_metric.update(syn_for_lpips, real_for_lpips)

        # === 5. Save Snapshots ===
        if not snapshots_saved and args.output_dir:
            # torchvision save_image handles (N, 1, H, W) and (N, 3, H, W) automatically
            save_image(
                synthetic_samples,
                fp=Path(args.output_dir) / "snapshots" / f"{epoch}_{data_iter_step}.png",
            )
            snapshots_saved = True

        # === 6. Save Samples for External FID (Generalized) ===
        if args.save_fid_samples and args.output_dir:
            # Convert to numpy: (N, C, H, W) -> (N, H, W, C)
            images_np = (
                (synthetic_samples * 255.0)
                .clip(0, 255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )
            
            image_dir = Path(args.output_dir) / "fid_samples"
            os.makedirs(image_dir, exist_ok=True)
            
            for batch_index, image_np in enumerate(images_np):
                image_path = (
                    image_dir
                    / f"{distributed_mode.get_rank()}_{data_iter_step}_{batch_index}.png"
                )
                
                # Auto-detect mode for PIL
                # image_np shape is (H, W, C)
                c_dim = image_np.shape[2]
                
                if c_dim == 1:
                    # Grayscale: Squeeze last dim -> (H, W) and use mode 'L'
                    img_pil = PIL.Image.fromarray(image_np[:, :, 0], "L")
                elif c_dim == 3:
                    # RGB: Use mode 'RGB'
                    img_pil = PIL.Image.fromarray(image_np, "RGB")
                else:
                    # Fallback for other channels (e.g., 4 channel RGBA or other)
                    # Try to save as is, or warn. Converting to RGB is safe fallback.
                    if c_dim == 4:
                         img_pil = PIL.Image.fromarray(image_np, "RGBA")
                    else:
                         # Force 3 channel
                         img_pil = PIL.Image.fromarray(image_np[:, :, :3], "RGB")

                img_pil.save(image_path)

        num_synthetic += batch_size

        if data_iter_step % PRINT_FREQUENCY == 0:
            logger.info(
                f"Evaluating [{data_iter_step}/{len(data_loader)}] "
                f"Generated [{num_synthetic}/{fid_samples}] "
                f"Current PSNR: {psnr_metric.compute():.4f}"
                f"Current SSIM: {ssim_metric.compute():.4f}"
                f"Current LPIPS: {lpips_metric.compute():.4f}"
            )

        if args.test_run:
            break

    # === Final Metric Computation ===
    logger.info("Computing final metrics...")
    
    # Compute final synchronized results
    final_psnr = float(psnr_metric.compute().detach().cpu())
    final_ssim = float(ssim_metric.compute().detach().cpu())
    final_lpips = float(lpips_metric.compute().detach().cpu())
    
    # Reset metrics to free memory
    psnr_metric.reset()
    ssim_metric.reset()
    lpips_metric.reset()
    
    results = {
        "psnr": final_psnr,
        "ssim": final_ssim,
        "lpips": final_lpips,
    }
    
    # === 保存指标文件 ===
    # 只有主进程负责写入文件，避免冲突
    if args.output_dir and distributed_mode.is_main_process():
        try:
            # 构造文件名，例如: eval_metrics_epoch_5.json
            metrics_save_path = Path(args.output_dir) / f"eval_metrics_epoch_{epoch}.json"
            
            with open(metrics_save_path, "w") as f:
                json.dump(results, f, indent=4)
                
            logger.info(f"Saved evaluation metrics to {metrics_save_path}")
        except Exception as e:
            logger.warning(f"Failed to save metrics json file: {e}")
    
    logger.info(f"Final Results for Epoch {epoch}: {results}")
    return results