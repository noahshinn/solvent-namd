import torch

from solvent_namd import computer

from typing import Dict, List, Optional


class Snapshot():
    def __init__(
        self,
        iteration: int,
        state: int,
        one_hot: torch.Tensor,
        one_hot_key: Dict,
        coords: torch.Tensor,
        energy: torch.Tensor,
        forces: torch.Tensor,
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
        self._iter: int = iteration
        self._state: int = state
        self._atom_types: List[str] = computer.one_hot_to_atom_type(one_hot, one_hot_key)
        self._coords: List[List[float]] = coords.tolist()
        self._energy: float = energy.item()
        self._forces: List[List[float]] = forces.tolist()
        # self._nacs = nacs.tolist()
        # self._socs = socs.tolist()

    def info_atom_types(self) -> List[str]:
        return self._atom_types

    def info_coords(self) -> List[List[float]]:
        return self._coords
