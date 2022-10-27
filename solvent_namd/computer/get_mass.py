"""
STATUS: NOT TESTED

"""

import torch
from ase.data import atomic_masses_iupac2016, atomic_numbers

from typing import Union, List


def get_mass(species: Union[str, List[str]]) -> torch.Tensor:
    """Returns a tensor of atomic masses"""
    return torch.cat([atomic_masses_iupac2016[atomic_numbers[s]] for s in species])
