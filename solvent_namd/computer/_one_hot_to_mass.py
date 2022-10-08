"""
STATUS: NOT TESTED

"""

import torch

from solvent_namd import utils

from typing import Dict


def one_hot_to_mass(one_hot: torch.Tensor, mass_key: Dict[int, float]) -> torch.Tensor:
    """
    Converts a one-hot atomic encoding to a mass tensor.

    Args:
        one_hot (torch.Tensor): One-hot atomic encoding.
        mass_key (Dict): keys: hash value -> values: atomic mass

    Returns:
        (torch.Tensor): A tensor of atomic masses.

    """
    mass = [mass_key[utils.hash_1d_tensor(atom)] for atom in one_hot]

    return torch.Tensor(mass)
