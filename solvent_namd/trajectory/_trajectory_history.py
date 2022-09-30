"""
STATUS: DEV

"""

import warnings

from solvent_namd.trajectory import Snapshot
from typing import List, Optional, NamedTuple


class SparseInfo(NamedTuple):
    atom_types: List[List[str]]
    coords: List[List[List[float]]]


class TrajectoryHistory:
    """
    A queue of molecular system snapshots with an optional max length.

    """
    def __init__(
        self,
        max_length: Optional[int]=None
    ) -> None:
        """
        Initializes an empty history of molecular system snapshots.

        Args:
            max_length (int | None): An optional max length to save memory.

        Returns:
            None

        """
        if max_length:
            self._max_len = max_length
        self._len = 0
        self._data: List[Snapshot] = []

    def add(self, s: Snapshot) -> None:
        """
        Adds a snapshot to the end of the history queue.

        Args:
            s (Snapshot): A molecular system snapshot.

        Returns:
            None

        """
        self._data.append(s)
        if self._max_len and self._len >= self._max_len:
            self._data.pop(0)
        else:
            self._len += 1

    def sparse_info(self) -> SparseInfo:
        """
        Returns atom type and coordinate data at corresponding indexes.

        Args:
            None

        Returns:
            atom_types (list(list(float))),
            coords (list(list(list(float)))): Of shape (N, K, 3) where N is the
                number of snapshots, K is the number of atoms per system, and 3
                for x, y, and z

        """
        if self._max_len:
            warnings.warn(f'max length of {self._max_len} is enforced, only returning last {self._len} snapshots.')
        atom_types = []
        coords = []
        for s in self._data:
            atom_types.extend([s.info_atom_types()])
            coords.extend([s.info_coords()])

        return SparseInfo(atom_types, coords)

    def all_info(self) -> List[Snapshot]:
        """
        Returns the complete information per snapshot.

        Args:
            None
        
        Returns:
            (list(Snapshot)): A list of molecular system snapshots.

        """
        if self._max_len:
            warnings.warn(f'max length of {self._max_len} is enforced, only returning last {self._len} snapshots.')
        return self._data
