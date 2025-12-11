# signal_image_dataset.py

import torch
import h5py
import numpy as np
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SignalImageDataset(Dataset):
    def __init__(self, data_path, transform=None, mode='train'):
        """
        data_path: 
            - 对于 CT 任务: 指向 .h5 文件的路径
            - 对于 LOL 任务: 指向包含 Train/Test 的根目录
        mode: 'train' 或 'test' (仅对 LOL 任务有效)
        """
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.mode = mode
        
        # 1. 判断任务类型
        if os.path.isfile(data_path) and data_path.endswith('.h5'):
            self.task_type = 'ct_h5'
            self._init_h5_dataset()
        elif os.path.isdir(data_path):
            self.task_type = 'lol_folder'
            self._init_folder_dataset()
        else:
            raise ValueError(f"Invalid data_path: {data_path}. Must be .h5 file or directory.")

    def _init_h5_dataset(self):
        # CT 任务初始化逻辑 (保持你原有的逻辑)
        print(f"[Dataset] Detected H5 file. Loading CT dataset from {self.data_path}...")
        self.data_key = 'images'
        self.condition_key = 'voltages'
        
        with h5py.File(self.data_path, 'r') as f:
            self.length = len(f[self.data_key])
            all_signals = np.array(f[self.condition_key])
        
        self.signal_mean = torch.from_numpy(all_signals.mean(axis=0)).float()
        self.signal_std = torch.from_numpy(all_signals.std(axis=0)).float()
        self.signal_std[self.signal_std == 0] = 1.0
        
        print(f"[Dataset] Size: {self.length}")
        print(f"[Dataset] Signal Mean (first 5): {self.signal_mean[:5]}")

    def _init_folder_dataset(self):
        # LOL 任务初始化逻辑
        print(f"[Dataset] Detected Directory. Loading LOL dataset from {self.data_path}...")
        
        # 假设目录结构: root/Train/Normal, root/Train/Low
        split = "Train" if self.mode == 'train' else "Test"
        root_dir = Path(self.data_path) / split
        
        self.path_high = root_dir / "Normal"
        self.path_low = root_dir / "Low"
        
        if not self.path_high.exists() or not self.path_low.exists():
             raise ValueError(f"LOL dataset structure error. Expected {self.path_high} and {self.path_low}")

        self.image_names = sorted([
            f for f in os.listdir(self.path_high) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        self.length = len(self.image_names)
        print(f"[Dataset] Size: {self.length} pairs")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.task_type == 'ct_h5':
            return self._getitem_h5(idx)
        else:
            return self._getitem_folder(idx)

    def _getitem_h5(self, idx):
        with h5py.File(self.data_path, 'r') as f:
            image_data = f[self.data_key][idx]
            signal_data = f[self.condition_key][idx]

        # 图像处理 (1通道)
        if self.transform:
            image = self.transform(image_data)
        else:
            image = torch.tensor(image_data, dtype=torch.float32)

        # 信号处理 (Vector)
        signal = torch.tensor(signal_data, dtype=torch.float32)
        signal_norm = (signal - self.signal_mean) / self.signal_std

        return {
            'data': image,           # [1, H, W]
            'condition': signal_norm # [208] Vector
        }

    def _getitem_folder(self, idx):
        img_name = self.image_names[idx]
        high_path = self.path_high / img_name
        low_path = self.path_low / img_name
        
        # 加载图片 (RGB 3通道)
        high_img = Image.open(high_path).convert("RGB") # Normal Light (Target)
        low_img = Image.open(low_path).convert("RGB")   # Low Light (Condition)

        # 强制 Resize 到 600x400 (或者你在 Transform 里做)
        # 注意: 600x400 不是 16 的倍数，建议 Resize 到 512x512 或 512x384
        # 这里为了演示，我们假设 transform 会处理 Resize
        # 如果没有 transform，手动转 tensor
        
        if self.transform:
            # Transform 需要能同时处理两张图，或者分别调用
            # 简单起见，假设 transform 是标准的 torchvision transforms
            # 必须保证两者做了一样的 Resize/Crop!
            # 实际项目中建议写一个 JointTransform
            high_img = self.transform(high_img)
            low_img = self.transform(low_img)
        else:
            # 简易 fallback
            to_tensor = transforms.ToTensor()
            high_img = to_tensor(high_img)
            low_img = to_tensor(low_img)

        return {
            'data': high_img,    # [3, H, W]
            'condition': low_img # [3, H, W] Image Condition
        }