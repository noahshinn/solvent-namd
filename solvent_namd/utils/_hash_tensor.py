"""
STATUS: NOT TESTED

"""

import torch


def hash_1d_tensor(x: torch.Tensor) -> int:
    """
    Hashes a 1d tensor to be stored as a key in a python dictionary.

    Args:
        x (torch.Tensor): Input tensor to be hashed. *Assumes a 1d tensor

    Returns:
        h (int): Hash value.

    """
    t = tuple(x.tolist())
    h = hash(t)

    return h
