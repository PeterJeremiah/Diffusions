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
                conditional = self.model(x, t, extra={"label": label})
                condition_free = self.model(x, t, extra={})
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

    # ======================================
    #FID指标
    #fid_metric = FrechetInceptionDistance(normalize=True).to(
    #    device=device, non_blocking=True
    #)
    #Inception Score (IS)
    #split=10 是标准设置
    #is_metric = InceptionScore(normalize=True).to(device=device, non_blocking=True)
    # PSNR
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device=device, non_blocking=True)
    # SSIM
    #ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device=device, non_blocking=True)
    # Perceptual Metric
    # net_type='vgg' 是最常用的感知距离设置
    #lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device=device, non_blocking=True)
    # ======================================

    num_synthetic = 0
    snapshots_saved = False
    if args.output_dir:
        (Path(args.output_dir) / "snapshots").mkdir(parents=True, exist_ok=True)
    #for data_iter_step, (samples, labels) in enumerate(data_loader):
    for data_iter_step, batch in enumerate(data_loader):
        #samples = samples.to(device, non_blocking=True)
        #labels = labels.to(device, non_blocking=True)
        samples = batch['data'].to(device, non_blocking=True)
        labels = batch['condition'].to(device, non_blocking=True)

        # 灰度图单通道，扩展为三通过，符合FID测试
        #samples_for_metric = samples
        #if samples.shape[1] == 1:
        #    samples_for_metrics = samples.repeat(1, 3, 1, 1)
        # =========================================


        #fid_metric.update(samples, real=True)

        if num_synthetic < fid_samples:
            cfg_scaled_model.reset_nfe_counter()
            if args.discrete_flow_matching:
                # Discrete sampling
                x_0 = (
                    torch.zeros(samples.shape, dtype=torch.long, device=device)
                    + MASK_TOKEN
                )
                if args.sym_func:
                    sym = lambda t: 12.0 * torch.pow(t, 2.0) * torch.pow(1.0 - t, 0.25)
                else:
                    sym = args.sym
                if args.sampling_dtype == "float32":
                    dtype = torch.float32
                elif args.sampling_dtype == "float64":
                    dtype = torch.float64

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
                # Continuous sampling
                x_0 = torch.randn(samples.shape, dtype=torch.float32, device=device)

                if args.edm_schedule:
                    time_grid = get_time_discretization(nfes=ode_opts["nfe"])
                else:
                    time_grid = torch.tensor([0.0, 1.0], device=device)

                synthetic_samples = solver.sample(
                    time_grid=time_grid,
                    x_init=x_0,
                    method=args.ode_method,
                    return_intermediates=False,
                    atol=ode_opts["atol"] if "atol" in ode_opts else 1e-5,
                    rtol=ode_opts["rtol"] if "atol" in ode_opts else 1e-5,
                    step_size=ode_opts["step_size"]
                    if "step_size" in ode_opts
                    else None,
                    label=labels,
                    cfg_scale=args.cfg_scale,
                )

                # Scaling to [0, 1] from [-1, 1]
                synthetic_samples = torch.clamp(
                    synthetic_samples * 0.5 + 0.5, min=0.0, max=1.0
                )
                synthetic_samples = torch.floor(synthetic_samples * 255)
            synthetic_samples = synthetic_samples.to(torch.float32) / 255.0




            # =============================================
            # 同样，将生成的单通道图转为 3 通道
            #if synthetic_samples.shape[1] == 1:
            #    synthetic_samples = synthetic_samples.repeat(1, 3, 1, 1)
            # =============================================

            logger.info(
                f"{samples.shape[0]} samples generated in {cfg_scaled_model.get_nfe()} evaluations."
            )
            if num_synthetic + synthetic_samples.shape[0] > fid_samples:
                synthetic_samples = synthetic_samples[: fid_samples - num_synthetic]
                
                # 同时也需要裁剪 real samples 以进行成对比较 (PSNR/SSIM)
                samples = samples[: synthetic_samples.shape[0]]



            # === 更新评估指标 ===
            # FID
            #fid_metric.update(synthetic_samples, real=False)
            # IS
            #is_metric.update(synthetic_samples)
            # PSNR
            psnr_metric.update(synthetic_samples, samples)
            # SSIM
            #ssim_metric.update(synthetic_samples, samples)
            # LPIPS
            #lpips_metric.update(synthetic_samples, samples)
            # ===========================


            num_synthetic += synthetic_samples.shape[0]
            if not snapshots_saved and args.output_dir:
                save_image(
                    synthetic_samples,
                    fp=Path(args.output_dir)
                    / "snapshots"
                    / f"{epoch}_{data_iter_step}.png",
                )
                snapshots_saved = True

            if args.save_fid_samples and args.output_dir:
                images_np = (
                    (synthetic_samples * 255.0)
                    .clip(0, 255)
                    .to(torch.uint8)
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .numpy()
                )
                for batch_index, image_np in enumerate(images_np):
                    image_dir = Path(args.output_dir) / "fid_samples"
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = (
                        image_dir
                        / f"{distributed_mode.get_rank()}_{data_iter_step}_{batch_index}.png"
                    )
                    PIL.Image.fromarray(image_np, "RGB").save(image_path)

        if not args.compute_fid:
            return {}

        #if data_iter_step % PRINT_FREQUENCY == 0:
        #    # Sync fid metric to ensure that the processes dont deviate much.
        #    gc.collect()
        #    running_fid = fid_metric.compute()
        #    logger.info(
        #        f"Evaluating [{data_iter_step}/{len(data_loader)}] samples generated [{num_synthetic}/{fid_samples}] running fid {running_fid}"
        #    )
        if data_iter_step % PRINT_FREQUENCY == 0:
            # Sync psnr metric to ensure that the processes dont deviate much.
            gc.collect()
            running_psnr = psnr_metric.compute()
            logger.info(
                f"Evaluating [{data_iter_step}/{len(data_loader)}] samples generated [{num_synthetic}/{fid_samples}] running psnr {running_psnr}"
            )

        if args.test_run:
            break

    # === 计算最终结果并返回字典 ===
    logger.info("Computing final metrics...")
    results = {
        #"fid": float(fid_metric.compute().detach().cpu()),
        #"inception_score_mean": float(is_metric.compute()[0].detach().cpu()), # IS 返回 (mean, std)
        #"inception_score_std": float(is_metric.compute()[1].detach().cpu()),
        "psnr": float(psnr_metric.compute().detach().cpu()),
        #"ssim": float(ssim_metric.compute().detach().cpu()),
        #"lpips": float(lpips_metric.compute().detach().cpu()), # LPIPS 越低越好
    }
    return results
