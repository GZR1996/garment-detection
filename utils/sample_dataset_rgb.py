import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    """ Sample dataset"""

    def __init__(self, sample_dir, csv_dir, transform=None):
        self.sample_dir = sample_dir
        self.sample_names = pd.read_csv(csv_dir)
        self.transform = transform

    def __len__(self):
        return int(len(self.sample_names))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        sample_names = self.sample_names.iloc[idx]
        samples_dict = {}
        for i, sample_name in enumerate(sample_names):
            sample_name = os.path.join(self.sample_dir, sample_name)
            sample = Image.open(sample_name)
            if self.transform:
                sample = self.transform(sample)
            samples_dict['x' + str(i+1)] = sample

        # sample_name = os.path.join(self.sample_dir, sample_names[0])
        # data = np.load(sample_name)
        # sample1 = np.expand_dims(data['image'], axis=0)
        # if self.transform:
        #     sample1 = self.transform(sample1)
        #
        # sample_name = os.path.join(self.sample_dir, sample_names[1])
        # data = np.load(sample_name)
        # sample2 = np.expand_dims(data['image'], axis=0)
        # if self.transform:
        #     sample2 = self.transform(sample2)
        #
        # sample_name = os.path.join(self.sample_dir, sample_names[2])
        # data = np.load(sample_name)
        # sample3 = np.expand_dims(data['image'], axis=0)
        # if self.transform:
        #     sample3 = self.transform(sample3)

        label = sample_names[0].replace('.npz\n', '').split('_')[:3]
        label = np.array([float(label[0]), float(label[1]), float(label[2])])
        samples_dict['label'] = label

        return samples_dict
