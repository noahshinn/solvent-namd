import torch


def kcal_to_hartree(x: torch.Tensor) -> torch.Tensor:
    """
    Converts kcals/mol to Hartrees.

    Args:
        x (torch.Tensor)

    Returns:
        (torch.Tensor)

    """
    return x / 627.503
