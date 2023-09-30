import torch


def hartree_to_ev(x: torch.Tensor) -> torch.Tensor:
    """
    Converts Hartrees to eVs.

    Args:
        x (torch.Tensor)

    Returns:
        (torch.Tensor)

    """
    return x * 27.2113961
