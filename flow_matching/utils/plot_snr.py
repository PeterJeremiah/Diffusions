# 文件路径: examples/image/plot_snr.py
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def get_log_snr_flow_matching(t):
    """
    Flow Matching / Optimal Transport Path (你的模型)
    路径: x_t = t * x_1 + (1 - t) * x_0
    (注意：这里假设 t=0 是噪声，t=1 是数据)
    Signal (alpha) = t
    Noise (sigma) = 1 - t
    SNR = (alpha / sigma)^2 = (t / (1-t))^2
    Log-SNR = 10 * log10(SNR)
    """
    t = np.clip(t, 1e-5, 1 - 1e-5) # 避免除以 0
    snr = (t / (1 - t)) ** 2
    return 10 * np.log10(snr)

def get_log_snr_dit(t):
    """
    DiT / Stable Diffusion (Linear Schedule)
    标准 Diffusion 通常 t=0 是数据，t=1 是噪声。
    为了与 Flow Matching 对齐，我们将 t 反转：t_fm = 1 - t_diff
    """
    # 模拟 Linear Schedule 的 Beta
    num_steps = 1000
    beta_start = 0.0001
    beta_end = 0.02
    betas = np.linspace(beta_start, beta_end, num_steps)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas)
    
    # 映射时间 t (0->1) 到 steps
    # FM t=0 (Noise) -> Diff t=1000
    # FM t=1 (Data)  -> Diff t=0
    t_idx = ((1 - t) * (num_steps - 1)).astype(int)
    alpha_bar = alphas_cumprod[t_idx]
    
    snr = alpha_bar / (1 - alpha_bar)
    return 10 * np.log10(snr)

def get_log_snr_sit(t):
    """
    SiT / iDDPM (Cosine Schedule)
    通常被认为比 Linear 更好
    """
    # 反转时间以匹配 FM
    t_diff = 1 - t
    s = 0.008
    f = np.cos(((t_diff + s) / (1 + s)) * np.pi / 2) ** 2
    alpha_bar = f / (np.cos(s / (1 + s) * np.pi / 2) ** 2)
    
    snr = alpha_bar / (1 - alpha_bar + 1e-8)
    return 10 * np.log10(snr)

def save_snr_plot(output_dir):
    """
    生成并保存对比图
    """
    if not os.path.exists(output_dir):
        return

    # 生成时间轴 (0 -> 1)
    t = np.linspace(0.001, 0.999, 1000)

    # 计算曲线
    fm_snr = get_log_snr_flow_matching(t)
    dit_snr = get_log_snr_dit(t)
    sit_snr = get_log_snr_sit(t)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(t, fm_snr, label='Flow Matching (Optimal Transport)', color='red', linewidth=2.5)
    plt.plot(t, dit_snr, label='DiT / Stable Diffusion (Linear)', color='blue', linestyle='--')
    plt.plot(t, sit_snr, label='SiT / iDDPM (Cosine)', color='green', linestyle='-.')

    # 装饰
    plt.title('SNR Schedule Comparison (Log Scale)', fontsize=16)
    plt.xlabel('Time Step $t$ (0=Noise $\\rightarrow$ 1=Data)', fontsize=14)
    plt.ylabel('Log-SNR (dB)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.axhline(y=0, color='black', linestyle=':', alpha=0.3) # SNR=1 分界线

    # 保存
    save_path = os.path.join(output_dir, 'snr_schedule_comparison.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"SNR Analysis plot saved to: {save_path}")