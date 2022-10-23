"""
STATUS: FINISHED

"""

import torch


def ev_to_hartree(x: torch.Tensor) -> torch.Tensor:
    """
    Converts eVs to Hartrees.

    Args:
        x (torch.Tensor)

    Returns:
        (torch.Tensor)

    """
    return x / 27.2113961
