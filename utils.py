import torch


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


def load_checkpoint(filename):
    """
    Load checkpoint
    :param filename:
    :return: dict of model, include parameters and states
    """
    return torch.load(filename)
