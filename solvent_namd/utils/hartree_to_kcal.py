"""
STATUS: NOT TESTED

"""

import torch


def hartree_to_kcal(x: torch.Tensor) -> torch.Tensor:
    """
    Converts Hartrees to kcals/mol.

    Args:
        x (torch.Tensor)

    Returns:
        (torch.Tensor)

    """
    return x * 627.503
