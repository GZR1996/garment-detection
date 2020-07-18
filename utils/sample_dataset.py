import os

import imageio
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    """ Sample dataset"""

    def __init__(self, sample_dir, csv_dir, transform=None):
        self.sample_dir = sample_dir
        self.labels = pd.read_csv(csv_dir)
        self.transform = transform

    def __len__(self):
        return int(len(self.labels)*0.2)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        label = np.array(self.labels.iloc[idx])
        sample_name = "{:.1f}_{:.1f}_{:.1f}_{:.0f}_{:.0f}.npz".format(label[0], label[1], label[2], label[3], label[4])
        sample_name = os.path.join(self.sample_dir, sample_name)
        data = np.load(sample_name)
        sample = np.expand_dims(data['image'], axis=0)

        if self.transform:
            sample = self.transform(sample)

        return {'sample': sample, 'label': list(self.labels.iloc[idx, :3])}
