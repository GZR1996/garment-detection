import os

import imageio
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class GarmentDataset(Dataset):
    """ Garment dataset"""

    def __init__(self, data_dir, data_type, csv_dir, transform=None):
        self.data_dir = data_dir
        self.data_type = data_type
        self.labels = pd.read_csv(csv_dir)
        self.transform = transform

    def __len__(self):
        return int(len(self.labels)*0.2)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        sample_names = np.array(self.labels.iloc[idx])
        print(sample_names)
        samples = []
        for i in range(1, 4):
            sample_name = os.path.join(self.data_dir, sample_names[i])
            sample = np.load(sample_name)
            sample = np.expand_dims(sample, axis=0)
            if self.transform:
                sample = self.transform(sample)
            samples.append(sample)

        label = sample_names[0].replace('.npz\n', '').split('_')[:3]
        label = np.array([float(label[0]), float(label[1]), float(label[2])])

        return {'samples': samples, 'label': label}
