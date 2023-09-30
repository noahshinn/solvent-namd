import torch


def bohr_to_angstrom(x: torch.Tensor) -> torch.Tensor:
    """
    Converts Bohrs to Angstroms.

    Args:
        x (torch.Tensor)

    Returns:
        (torch.Tensor)

    """
    return x * 0.529177249
