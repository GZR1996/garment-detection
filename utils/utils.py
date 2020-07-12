import os

import torch
import numpy as np
import matplotlib.pyplot as plt

IMAGE_SIZE = 3
DATA_SIZE = 1
LATENT_SIZE = 32
BEST_FILENAME = 'best.txt'


def save_checkpoint(state, is_best, filename, best_filename):
    """
    Save checkpoint and store the best checkpoint
    :param state: dict of model, include parameters and states
    :param is_best: boolean, mark if the checkpoint is the best
    :param filename: filename to save
    :param best_filename: the filename of best checkpoint
    :return: None
    """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)


def save_image(sample_dir, data, labels):
    """

    :param sample_dir:
    :param data:
    :param label:
    :return:
    """
    for image, label in zip(data, labels):
        image_dir = os.path.join(sample_dir, "{:.1f}_{:.1f}_{:.1f}_{:.0f}_{:.0f}".format(label[0], label[1],
                                                                                         label[2], label[3], label[4]))
        np.savez_compressed(image_dir, image=image[0], label=label)


a = np.load('/home/zirun/project/python/garment-detection/simulation/data/sample/40.0_0.1_2.0_1_0.npz')
print(a['image'])
plt.imshow(a['image'])
plt.show()