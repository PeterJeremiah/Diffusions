# 文件名: visualize_h5_save.py

import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import os

# --- 配置 ---
NUM_SAMPLES = 10  # 随机抽取图像数量
SAVE_DIR = "/mnt/shared-storage-user/yangmingyuan/h5_visualizations"  # 保存的文件夹

# --- 检查命令行参数 ---
if len(sys.argv) != 2:
    print("用法: python visualize_h5_save.py /path/to/file.h5")
    sys.exit(1)

file_path = sys.argv[1]

# 创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 打开 H5 文件 ---
try:
    with h5py.File(file_path, 'r') as f:
        print(f"--- H5 文件: {file_path} ---")
        print("顶层键:", list(f.keys()))
        
        images = f['images']
        masks = f['masks']
        voltages = f['voltages']
        
        print(f"images shape: {images.shape}, dtype: {images.dtype}")
        print(f"masks shape: {masks.shape}, dtype: {masks.dtype}")
        print(f"voltages shape: {voltages.shape}, dtype: {voltages.dtype}")
        
        # 随机选择几张图像进行可视化
        indices = random.sample(range(images.shape[0]), min(NUM_SAMPLES, images.shape[0]))
        
        for i, idx in enumerate(indices):
            img = images[idx]
            mask = masks[idx]
            voltage = voltages[idx]
            
            plt.figure(figsize=(12,4))
            
            # 显示原图
            plt.subplot(1,3,1)
            plt.imshow(img, cmap='gray')
            plt.title(f"Image idx={idx}")
            plt.axis('off')
            
            # 显示 mask
            plt.subplot(1,3,2)
            plt.imshow(mask, cmap='gray')
            plt.title("Mask")
            plt.axis('off')
            
            # 显示 voltage 条件向量
            plt.subplot(1,3,3)
            plt.plot(voltage)
            plt.title("Voltage (condition)")
            plt.tight_layout()
            
            # 保存图像
            save_path = os.path.join(SAVE_DIR, f"sample_{idx}.png")
            plt.savefig(save_path)
            plt.close()
            
            print(f"Saved visualization: {save_path}")

except Exception as e:
    print("打开 H5 文件失败:", e)
