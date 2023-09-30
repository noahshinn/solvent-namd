import torch


def angstrom_to_bohr(x: torch.Tensor) -> torch.Tensor:
    """
    Converts Angstroms to Bohrs.

    Args:
        x (torch.Tensor)

    Returns:
        (torch.Tensor)

    """
    return x / 0.529177249
