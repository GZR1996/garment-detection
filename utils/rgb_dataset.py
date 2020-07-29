import os

import imageio
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class GarmentDataset(Dataset):
    """ Garment dataset"""

    def __init__(self, rgb_dir, data_dir, data_type, csv_dir, transform=None):
        self.rgb_dir = rgb_dir
        self.data_dir = data_dir
        self.data_type = data_type
        self.labels = pd.read_csv(csv_dir)
        self.transform = transform

    def __len__(self):
        return int(len(self.labels))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        label = np.array(self.labels.iloc[idx])
        sample = None

        if self.data_type == 'rgb':
            img_name = "{}_{}_{}_{:.0f}_{:.0f}.jpg".format(label[0], label[1], label[2], label[3], label[4])
            img_name = os.path.join(self.rgb_dir, img_name)
            img = Image.open(img_name)
            sample = img
        elif self.data_type in ['raw_depth', 'depth', 'segmentation']:
            data_name = "{}_{}_{}_{:.0f}_{:.0f}.npz".format(label[0], label[1], label[2], label[3], label[4])
            data_name = os.path.join(self.data_dir, data_name)
            data = np.load(data_name)
            sample = np.expand_dims(data[self.data_type], axis=0)

        if self.transform:
            sample = self.transform(sample)

        return {'sample': sample, 'label': label}
