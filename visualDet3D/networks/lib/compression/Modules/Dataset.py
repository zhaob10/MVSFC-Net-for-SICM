import h5py
import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class H5Dataset(Dataset):
    def __init__(self, h5_file):
        super(H5Dataset, self).__init__()
        self.h5 = h5py.File(name=h5_file, mode='r')
        self.file_list = list(self.h5.keys())

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        frames = self.h5[self.file_list[idx]]
        frames = np.array(frames)
        frames = torch.from_numpy(frames) / 255.0  # scale to [0, 1]
        frames = frames.permute(0, 3, 1, 2).contiguous()
        return frames
