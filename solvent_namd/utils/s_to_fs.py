"""
STATUS: FINISHED

"""

import torch

from typing import Union


def s_to_fs(x: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
    """
    Converts s to fs.

    Args:
        x (float | torch.Tensor)

    Returns:
        (float | torch.Tensor)

    """
    return x / 1e-15
