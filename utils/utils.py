import os

import torch
import imageio

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


def save_image(sample_dir, data, label):
    """

    :param sample_dir:
    :param data:
    :param label:
    :return:
    """
    for image in data:
        image_dir = os.path.join(sample_dir, "{:.1f}_{:.1f}_{:.1f}_{:.0f}_{:.0f}".format(label[0], label[1],
                                                                                         label[2], label[3], label[4]))
        imageio.imwrite(image_dir, image)
