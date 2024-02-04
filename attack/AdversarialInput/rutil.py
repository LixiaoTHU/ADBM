import os
import torch
import torch.nn as nn
import numpy as np
import random
import logging
import functools
import sys


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print




def random_seed(seed=0, rank=0):
    seed = seed + rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_random_states():
    state = np.random.get_state()
    torch_state = torch.get_rng_state()
    rstate = random.getstate()
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state_all()
    else:
        cuda_state = None
    return state, torch_state, rstate, cuda_state

def set_random_states(states):
    state, torch_state, rstate, cuda_state = states
    np.random.set_state(state)
    torch.set_rng_state(torch_state)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
    random.setstate(rstate)

@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name='', default_level=logging.INFO, save_log = True):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(default_level)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(default_level)
        console_handler.setFormatter(
            logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # create file handlers
    if save_log:
        torch.distributed.barrier()
        file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}.txt'), mode='a')
        file_handler.setLevel(default_level)
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)

    return logger


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        def normalize_fn(tensor, mean, std):
            """Differentiable version of torchvision.functional.normalize"""
            # here we assume the color channel is in at dim=1
            mean = mean[None, :, None, None]
            std = std[None, :, None, None]
            return tensor.sub(mean).div(std)

        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)