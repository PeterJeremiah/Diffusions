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
    def __init__(self, data_path, transform=None, mode='train', dataset_name=None):
        """
        data_path: 数据路径
        mode: 'train' 或 'test'
        dataset_name: 显式指定任务名称 ('imagenet', 'lol', 'euvp')
        """
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.mode = mode
        self.dataset_name = dataset_name
        
        # ============================================================
        # === 核心修改：根据 dataset_name 进行显式任务分发 ===
        # ============================================================
        
        if self.dataset_name == 'imagenet':
            # --- CT 任务 (H5) ---
            self.task_type = 'ct_h5'
            self._init_h5_dataset()
            
        elif self.dataset_name == 'lol':
            # --- LOL 任务 (Folder) ---
            # 完全保留原有逻辑，不做任何额外修改
            self.task_type = 'lol_folder'
            self._init_folder_dataset()
            
        elif self.dataset_name == 'euvp':
            # --- EUVP 任务 (Folder) ---
            # 仿照 LOL 格式，但走独立分支
            self.task_type = 'euvp_folder'
            self._init_euvp_dataset()
            
        else:
            # 兼容旧代码：如果没有传 dataset_name，尝试自动推断 (Fallback)
            if os.path.isfile(data_path) and data_path.endswith('.h5'):
                self.task_type = 'ct_h5'
                self._init_h5_dataset()
            elif os.path.isdir(data_path):
                self.task_type = 'lol_folder'
                self._init_folder_dataset()
            else:
                raise ValueError(f"Invalid data_path: {data_path}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 根据 task_type 分发到具体的读取函数
        if self.task_type == 'ct_h5':
            return self._getitem_h5(idx)
        elif self.task_type == 'lol_folder':
            return self._getitem_folder(idx)
        elif self.task_type == 'euvp_folder':
            return self._getitem_euvp(idx)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    # ==========================================
    # 1. CT 任务逻辑 (保持原样)
    # ==========================================
    def _init_h5_dataset(self):
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

    def _getitem_h5(self, idx):
        with h5py.File(self.data_path, 'r') as f:
            image_data = f[self.data_key][idx]
            signal_data = f[self.condition_key][idx]

        if self.transform:
            image = self.transform(image_data)
        else:
            image = torch.tensor(image_data, dtype=torch.float32)

        signal = torch.tensor(signal_data, dtype=torch.float32)
        signal_norm = (signal - self.signal_mean) / self.signal_std

        return {
            'data': image, 
            'condition': signal_norm
        }

    # ==========================================
    # 2. LOL 任务逻辑 (保持原样，不做任何 Resize 修改)
    # ==========================================
    def _init_folder_dataset(self):
        print(f"[Dataset] Loading LOL dataset from {self.data_path}...")
        
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

    def _getitem_folder(self, idx):
        img_name = self.image_names[idx]
        high_path = self.path_high / img_name
        low_path = self.path_low / img_name
        
        high_img = Image.open(high_path).convert("RGB") 
        low_img = Image.open(low_path).convert("RGB")   
        
        # [关键] 这里完全保留你原始代码的逻辑
        # 不进行强制 Resize，完全依赖传入的 transform
        
        if self.transform:
            high_img = self.transform(high_img)
            low_img = self.transform(low_img)
        else:
            to_tensor = transforms.ToTensor()
            high_img = to_tensor(high_img)
            low_img = to_tensor(low_img)

        return {
            'data': high_img,    
            'condition': low_img 
        }

    # ==========================================
    # 3. [新增] EUVP 任务逻辑 (独立分支)
    # ==========================================
    def _init_euvp_dataset(self):
        print(f"[Dataset] Loading EUVP dataset from {self.data_path}...")
        
        split = "Train" if self.mode == 'train' else "Test"
        root_dir = Path(self.data_path) / split
        
        self.path_truth = root_dir / "Truth"         
        self.path_distorted = root_dir / "Distorted" 
        
        if not self.path_truth.exists() or not self.path_distorted.exists():
             raise ValueError(f"EUVP dataset structure error. Expected {self.path_truth} and {self.path_distorted}")

        self.image_names = sorted([
            f for f in os.listdir(self.path_truth) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        
        self.length = len(self.image_names)
        print(f"[Dataset] EUVP Size: {self.length} pairs")

    def _getitem_euvp(self, idx):
        img_name = self.image_names[idx]
        
        # === [修改点] 使用新的路径变量 ===
        truth_path = self.path_truth / img_name
        distorted_path = self.path_distorted / img_name
        
        truth_img = Image.open(truth_path).convert("RGB")
        distorted_img = Image.open(distorted_path).convert("RGB")
        
        # 依赖外部 transform 处理 (resize/totensor/normalize)
        if self.transform:
            truth_img = self.transform(truth_img)
            distorted_img = self.transform(distorted_img)
        else:
            to_tensor = transforms.ToTensor()
            truth_img = to_tensor(truth_img)
            distorted_img = to_tensor(distorted_img)

        return {
            'data': truth_img,      # Truth (Target)
            'condition': distorted_img # Distorted (Condition)
        }