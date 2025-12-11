# 文件名: check_dims.py

import torch
from torch.utils.data import DataLoader
import sys
import os

# --- 配置 ---
# 确保 H5 路径正确
H5_PATH = "/mnt/shared-storage-user/yangmingyuan/val.h5" 
BATCH_SIZE = 128 # 用一个小批量测试
# ---

# 1. 导入您自己的模块
# (假设此脚本与 train.py 在同一目录)
try:
    from signal_image_dataset import SignalImageDataset
    # 假设 transform 脚本在 'training' 文件夹中
    from training.data_transform import get_train_transform 
except ImportError as e:
    print(f"导入失败: {e}")
    print("请确保 check_dims.py 与 train.py 在同一目录下")
    print("并且 signal_image_dataset.py 也在该目录中")
    sys.exit(1)

if not os.path.exists(H5_PATH):
    print(f"错误: 找不到 H5 文件: {H5_PATH}")
    sys.exit(1)

print("--- 维度检查 Demo ---")

# 2. 初始化
print("正在初始化 Transform 和 Dataset...")
transform = get_train_transform()
dataset = SignalImageDataset(h5_path=H5_PATH, transform=transform)

# 3. 创建一个简单的 DataLoader
# (不使用 DistributedSampler，仅用于测试)
data_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0 # 设为 0 避免多进程问题
)

print("DataLoader 创建成功。正在获取一个批次...")

try:
    # 4. 获取一个批次
    batch = next(iter(data_loader))
    
    images = batch['data']
    condition = batch['condition']

    print("\n--- 检查结果 ---")
    print(f"批次大小 (Batch Size): {images.shape[0]}")
    
    # 5. 打印图像维度
    print(f"Images ('data') 维度: {images.shape}")
    print(f"  -> 格式应为 (Batch_Size, Channels, Height, Width)")
    
    # 6. 打印条件维度
    print(f"Condition ('condition') 维度: {condition.shape}")
    print(f"  -> 格式应为 (Batch_Size, Cond_Dim)")

    if len(condition.shape) == 2:
        cond_dim = condition.shape[1]
        print(f"\n***************************************")
        print(f"!! 您的 'cond_dim' 是: {cond_dim} !!")
        print(f"!! 请在 models/model_configs.py 中使用这个值 !!")
        print(f"***************************************")
    else:
        print(f"\n!! 警告: 您的条件维度不是 2 维 (Batch_Size, Cond_Dim)")
        print("请检查您的 H5 文件和 signal_image_dataset.py")

except Exception as e:
    print(f"\n--- 错误 ---")
    print(f"获取数据失败: {e}")
    print("请检查 signal_image_dataset.py 中的 __getitem__ 是否正确返回了 {'data': ..., 'condition': ...}")