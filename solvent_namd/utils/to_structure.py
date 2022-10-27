"""
STATUS: DEV

"""

from ase import Atoms

import torch
from typing import List


def to_structure(species: List[str], coords: torch.Tensor) -> Atoms:
    """Yields an ase.Atoms object"""
    return Atoms(species, coords) 
