import pandas as pd
import torch
import os
import imageio
import numpy as np
from torch.utils.data import Dataset


class GarmentDataset(Dataset):
    """ Garment dataset"""

    def __init__(self, data_dir, data_type, csv_dir, transform=None):
        self.data_dir = data_dir
        self.data_type = data_type
        self.labels = pd.read_csv(csv_dir)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        label = self.labels.iloc[idx]

        data_name = "{}_{}_{}_{:.0f}_{:.0f}.npz".format(label[0], label[1], label[2], label[3], label[4])
        data_name = os.path.join(self.data_dir, data_name)
        data = np.load(data_name)
        sample = data[self.data_type]

        if self.transform:
            sample = self.transform(sample)

        return sample
