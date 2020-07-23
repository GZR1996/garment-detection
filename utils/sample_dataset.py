import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    """ Sample dataset"""

    def __init__(self, sample_dir, csv_dir, transform=None):
        self.sample_dir = sample_dir
        self.sample_names = pd.read_csv(csv_dir)
        self.transform = transform

    def __len__(self):
        return int(len(self.sample_names) * 0.2)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        sample_names = self.sample_names.iloc[idx]
        samples = []
        for sample_name in sample_names:
            sample_name = os.path.join(self.sample_dir, sample_name)
            print(sample_names)
            data = np.load(sample_name)
            sample = np.expand_dims(data['image'], axis=0)
            if self.transform:
                sample = self.transform(sample)
            samples.append(samples)

        label = sample_names[0].replace('.npz\n', '').split('_')[:3]
        label = np.array([float(label[0]), float(label[1]), float(label[2])])

        return {'sample': np.array(sample), 'label': label}
