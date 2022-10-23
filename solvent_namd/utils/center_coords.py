"""
STATUS: DEV

"""

import torch


def center_coords(coords: torch.Tensor) -> torch.Tensor:
    center = coords.mean(dim=0)
    return coords - center
