"""
STATUS: DEV

"""

from solvent_namd import trajectory
from typing import List, Optional


class TrajectoryHistory:
    """
    A queue of molecular system snapshots with an optional max length.

    """
    def __init__(
        self,
        max_length: Optional[int]=5
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
        self._data: List[trajectory.Snapshot] = []

    def add(self, s: trajectory.Snapshot) -> None:
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
