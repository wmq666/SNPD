import torch
import numpy as np
from torch.utils.data import Dataset


class nr2n_dataset(Dataset):
    def __init__(self, file1, file2, wave_number):
        self.data1 = np.load(file1)
        self.data2 = np.load(file2)
        self.wave_number = wave_number

    def __len__(self):
        return min(len(self.data1), len(self.data2))   # 返回两个数据集中较小的数据点数量

    def __getitem__(self, idx):
        noise1 = torch.tensor(self.data1[idx, :], dtype=torch.float32).view(1, self.wave_number)
        noise2 = torch.tensor(self.data2[idx, :], dtype=torch.float32).view(1, self.wave_number)
        return noise1, noise2
