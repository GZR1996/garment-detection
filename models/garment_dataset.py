import pandas as pd
import torch
import os
import imageio
import numpy as np
from torch.utils.data import Dataset


class GarmentDataset(Dataset):
    """ Garment dataset"""

    def __init__(self, rgb_dir, data_dir, csv_dir, transform=None):
        self.rgb_dir = rgb_dir
        self.data_dir = data_dir
        self.labels = pd.read_csv(csv_dir)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        label = self.labels.iloc[idx]
        print(label[0], label[1], label[2], int(label[3]), int(label[4]))
        img_name = "{}_{}_{}_{:.0f}_{:.0f}.png".format(label[0], label[1], label[2], label[3], label[4])
        img_name = os.path.join(self.rgb_dir, img_name)
        img = imageio.read(img_name)

        data_name = "{}_{}_{}_{:.0f}_{:.0f}.npz".format(label[0], label[1], label[2], label[3], label[4])
        data_name = os.path.join(self.data_dir, data_name)
        data = np.load(data_name)

        sample = {'rgb': img, 'raw_depth': data['raw_depth'], 'depth': data['depth'],
                  'segmentation': data['segmentation']}

        if self.transform:
            sample = self.transform(sample)

        return sample



