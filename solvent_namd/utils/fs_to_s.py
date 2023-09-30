import torch

from typing import Union


def fs_to_s(x: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
    """
    Converts fs to s.

    Args:
        x (float | torch.Tensor)

    Returns:
        (float | torch.Tensor)

    """
    return x * 1e-15
