import torch

from solvent_namd import logger
from typing import List, Optional


class Snapshot():
    def __init__(
        self,
        iteration: int,
        state: int,
        atom_strings: List[str],
        coords: torch.Tensor,
        velo: torch.Tensor,
        forces: torch.Tensor,
        energies: torch.Tensor,
        nacs: Optional[torch.Tensor]=None,
        socs: Optional[torch.Tensor]=None
    ) -> None:
        """
        Stores information about a molecular system at a moment in time.

        Args:
            iteration (int): Iteration number within its trajectory
            state (int): Electronic state of which the molecular system is populating.
            one_hot (torch.Tensor): A one-hot tensor representing atom types.
            one_hot_key (torch.Tensor): The key of which to decode one-hot tensor. 
            coords (torch.Tensor): Positions of atomic coordinates in Angstroms.
            energy (torch.Tensor): The total potential energy of the molecular system.
            forces (torch.Tensor): Forces with respect to direction for every atom
                in the molecular system.
            nacs (torch.Tensor): NOT IMPLEMENTED
            socs (torch.Tensor): NOT IMPLEMENTED
        
        Returns:
            None

        """
        self._iteration = iteration
        self._state = state
        self._atom_strings = atom_strings
        self._coords = coords
        self._velo = velo
        self._forces = forces 
        self._energies = energies
        self._nacs = nacs
        self._socs = socs 

    def log(self, logger: logger.TrajLogger) -> None:
        logger.log_step(
            coords=self._coords,
            velo=self._velo,
            forces=self._forces,
            energies=self._energies,
            state=self._state
        )
