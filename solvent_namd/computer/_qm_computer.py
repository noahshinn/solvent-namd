"""
STATUS: DEV

"""

import torch

from typing import List
from solvent_namd.types import SPEnergiesForces


class QMComputer:
    def __init__(self) -> None:
        NotImplemented()

    def _energy_calculation(self, species: List[str], coords: torch.Tensor) -> torch.Tensor:
        e = ...
        return e # type: ignore

    def _force_calculation(
            self,
            species: List[str],
            coords: torch.Tensor,
            e: torch.Tensor
        ) -> torch.Tensor:
        f = ...
        return f # type: ignore

    def sp(self, species: List[str], coords: torch.Tensor) -> SPEnergiesForces:
        e = self._energy_calculation(species=species, coords=coords)
        f = self._force_calculation(species=species, coords=coords, e=e)
        return SPEnergiesForces(e, f)
