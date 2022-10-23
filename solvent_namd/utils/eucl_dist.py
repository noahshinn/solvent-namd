"""
STATUS: DEV

"""

import torch


def eucl_dist(a1: torch.Tensor, a2: torch.Tensor) -> torch.Tensor:
    return (a1 - a2).pow(2).sum().sqrt()
