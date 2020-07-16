import os

import torch
import numpy as np

from functools import partial

IMAGE_SIZE = 3
DATA_SIZE = 1
LATENT_SIZE = 1024
BEST_FILENAME = 'best.pth'


def save_checkpoint(loss_state, best_state, is_best, filename, best_filename):
    """
    Save vae and store the best vae
    :param loss_state: loss of each epoch
    :param best_state: dict of model, include parameters and states
    :param is_best: boolean, mark if the vae is the best
    :param filename: filename to save
    :param best_filename: the filename of best vae
    :return: None
    """
    torch.save(loss_state, filename)
    if is_best:
        torch.save(best_state, best_filename)


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
        np.savez_compressed(image_dir, image=image, label=label)


class EarlyStopping(object):
    """
    Gives a criterion to stop training when a given metric is not
    improving anymore
    Args:
        mode (str): One of `min`, `max`. In `min` mode, training will
            be stopped when the quantity monitored has stopped
            decreasing; in `max` mode it will be stopped when the
            quantity monitored has stopped increasing. Default: 'min'.
        patience (int): Number of epochs with no improvement after
            which training is stopped. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only stop learning after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
    """

    def __init__(self, mode='min', patience=10, threshold=1e-4, threshold_mode='rel'):
        self.patience = patience
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        """ Updates early stopping state """
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

    @property
    def stop(self):
        """ Should we stop learning? """
        return self.num_bad_epochs > self.patience

    def _cmp(self, mode, threshold_mode, threshold, a, best):  # pylint: disable=R0913, R0201
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return a < best * rel_epsilon

        elif mode == 'min' and threshold_mode == 'abs':
            return a < best - threshold

        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return a > best * rel_epsilon

        return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = float('inf')
        else:  # mode == 'max':
            self.mode_worse = (-float('inf'))

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)

    def state_dict(self):
        """ Returns early stopping state """
        return {key: value for key, value in self.__dict__.items() if key != 'is_better'}

    def load_state_dict(self, state_dict):
        """ Loads early stopping state """
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold,
                             threshold_mode=self.threshold_mode)
