import torch

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