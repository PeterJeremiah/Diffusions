#import torch
#from torch.utils.data import Dataset
#import h5py
#import numpy as np
#
#class SignalImageDataset(Dataset):
#    def __init__(self, h5_path, use_mask=False):
#        super().__init__()
#        self.h5_path = h5_path
#        self.use_mask = use_mask
#
#        with h5py.File(h5_path, 'r') as f:
#            self.images = np.array(f['images'])       # (N, 256, 256)
#            self.voltages = np.array(f['voltages'])   # (N, 208)
#            if use_mask:
#                self.masks = np.array(f['masks'])     # (N, 32, 32)
#
#        # 归一化到 [0, 1]
#        self.images = self.images.astype(np.float32) / 255.0
#        self.voltages = self.voltages.astype(np.float32)
#
#    def __len__(self):
#        return len(self.images)
#
#    def __getitem__(self, idx):
#        img = torch.tensor(self.images[idx], dtype=torch.float32).#unsqueeze(0)  # shape (1, 256, 256)
#        cond = torch.tensor(self.voltages[idx], dtype=torch.float32)  # #shape (208,)
#
#        if self.use_mask:
#            mask = torch.tensor(self.masks[idx], dtype=torch.float32).#unsqueeze(0)
#            return {"img": img, "cond": cond, "mask": mask}
#        else:
#            return {"img": img, "cond": cond}





#####################################################################
#2.0
#import torch
#import h5py
#from torch.utils.data import Dataset
## 注意：您可能需要从 train.py 中查看 get_train_transform() 到底返回了什么
## 这里我们假设它是一个标准的 torchvision transform
## from torchvision import transforms 
#
#class SignalImageDataset(Dataset):
#    def __init__(self, h5_path, transform=None):
#        self.h5_file = h5py.File(h5_path, 'r')
#        # 假设 H5 文件中有 'signals' 和 'images' 两个数据集
#        self.signals = self.h5_file['voltages']
#        self.images = self.h5_file['images']
#        self.transform = transform
#        self.length = len(self.signals)
#
#    def __len__(self):
#        return self.length
#
#    def __getitem__(self, idx):
#        signal_data = self.signals[idx]
#        image_data = self.images[idx]
#        
#        # 转换图像 (例如: 归一化, ToTensor)
#        # transform 是必须的！
#        if self.transform:
#            image = self.transform(image_data)
#        else:
#            # 如果没有 transform，至少要转成 Tensor
#            image = torch.tensor(image_data, dtype=torch.float32)
#
#        # 转换信号
#        signal = torch.tensor(signal_data, dtype=torch.float32)
#
#        # !! 关键：返回一个字典
#        # 这将使训练循环中的修改变得容易
#        return {
#            'data': image,       # 目标图像
#            'condition': signal  # 条件 (一维信号)
#        }
#
#    def close(self):
#        self.h5_file.close()


################################################################
#3.0
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset

class SignalImageDataset(Dataset):
    def __init__(self, h5_path, transform=None, data_key='images', condition_key='voltages'):
        super().__init__()
        self.h5_path = h5_path
        self.transform = transform
        self.data_key = data_key
        self.condition_key = condition_key
        
        # 读取数据用于统计
        print(f"Loading dataset from {h5_path} for statistics...")
        with h5py.File(h5_path, 'r') as f:
            self.length = len(f[self.data_key])
            # 加载所有信号计算均值和方差 (5000数据量很小，内存完全够)
            all_signals = np.array(f[self.condition_key]) 
        
        # 计算统计量
        self.signal_mean = torch.from_numpy(all_signals.mean(axis=0)).float()
        self.signal_std = torch.from_numpy(all_signals.std(axis=0)).float()
        
        # 防止除以0
        self.signal_std[self.signal_std == 0] = 1.0
        
        print(f"Dataset Size: {self.length}")
        print(f"Signal Mean (first 5): {self.signal_mean[:5]}")
        print(f"Signal Std (first 5): {self.signal_std[:5]}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 在 getitem 中打开文件，支持多进程 worker
        with h5py.File(self.h5_path, 'r') as f:
            image_data = f[self.data_key][idx]
            signal_data = f[self.condition_key][idx]

        # 1. 处理图像
        if self.transform:
            image = self.transform(image_data)
        else:
            image = torch.tensor(image_data, dtype=torch.float32)
            # 假设原始数据是 0-255，需要转为 [0, 1]
            # 如果 transform 里做了，这里就不用做
            # image = image / 255.0 

        # 2. 处理信号 (关键修正！！！)
        signal = torch.tensor(signal_data, dtype=torch.float32)
        signal_norm = (signal - self.signal_mean) / self.signal_std

        return {
            'data': image,       # [C, H, W]
            'condition': signal_norm # [208] (Normalized)
        }