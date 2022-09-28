"""
STATUS: DEV

"""

import torch


class NAMD():
    _model: torch.nn.Module
    _ntraj: int
    _nsteps: int
    _prop_duration: float
    _delta_t: float
    _nsteps: int

    def __init__(
        self,
        model: torch.nn.Module,
        ntraj: int,
        prop_duration: float,
        delta_t: float,
        init_cond: torch.Tensor
    ) -> None:
        """
        Manages all trajectory propagations.

        Args:
            model (torch.nn.Module): A trained and loaded neural network
                model.
            ntraj (int): The number of trajectories to propagate.
            prop_duration (float): The max duration of a trajectory.
            delta_t (float): The duration between steps in a trajectory.
            init_cond (torch.Tensor): Initial starting conditions.

        Returns:
            (None)

        """
        self._model = model
        self._ntraj = ntraj
        self._prop_duration = prop_duration
        self._delta_t = delta_t
        self._nsteps = int(prop_duration / delta_t)

    def run(self) -> None:
        for i in range(self._ntraj):
            traj = ...
            for step in range(self._nsteps):
                traj.propagate() # type: ignore
                if not traj.status(): # type: ignore
                    NotImplemented()
                else:
                    NotImplemented()
