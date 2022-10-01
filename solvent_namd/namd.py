"""
STATUS: DEV

"""

import os
import torch
import inspect

from solvent_namd import utils, logger, trajectory

from typing import Dict, Type, TypeVar


T = TypeVar('T', bound='NAMD')


class NAMD():
    _ncores: int
    _model: torch.nn.Module
    _ntraj: int
    _nsteps: int
    _prop_duration: float
    _delta_t: float
    _natoms: int
    _nstates: int
    _nsteps: int
    _init_cond: torch.Tensor
    _log_dir: str
    _model_name: str
    _description: str

    def __init__(
            self,
            ncores: int,
            model: torch.nn.Module,
            ntraj: int,
            prop_duration: float,
            delta_t: float,
            natoms: int,
            nstates: int,
            init_cond: torch.Tensor,
            atom_types: torch.Tensor,
            log_dir: str = 'out',
            model_name: str = 'Unnamed',
            description: str = 'No description'
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
        assert init_cond.size(dim=0) == ntraj

        self._ncores = ncores
        self._model = model
        self._ntraj = ntraj
        self._prop_duration = prop_duration
        self._delta_t = delta_t
        self._natoms = natoms
        self._nstates = nstates 
        self._init_cond = init_cond
        self._atom_types = atom_types
        self._nsteps = int(prop_duration / delta_t)
        self._log_dir = log_dir
        self._model_name = model_name
        self._description = description

    @classmethod
    def deserialize(cls: Type[T], d: Dict) -> T:
        params = inspect.getfullargspec(NAMD.__init__)
        for k in d:
            if not k in params:
                raise utils.InvalidInputError(f'input `{k}` was given but is not a valid input')
        for p in params:
            if not p in d:
                raise utils.InvalidInputError(f'input `{p}` was not given in input')
        return cls(**d)

    # TODO: deploy on all cores/threads
    def run(self) -> None:
        nterminated = 0
        lg = logger.NAMDLogger(
            f=os.path.join(self._log_dir, 'traj-all.log'),
            ntraj=self._ntraj,
            delta_t=self._delta_t,
            nsteps=self._nsteps,
            ncores=self._ncores,
            description=self._description,
            model_name=self._model_name,
            natoms=self._natoms,
            nstates=self._nstates
        )
        for i in range(self._ntraj):
            # TODO: send initial conditions
            traj_id = f'traj-{i}'
            traj_lg = logger.TrajLogger(
                f=os.path.join(self._log_dir, f'{traj_id}.log'),
                traj=i,
                ntraj=self._ntraj,
                delta_t=self._delta_t,
                nsteps=self._nsteps,
                atom_types=self._atom_types,
                nstates=self._nstates
            )
            traj_lg.log_header()
            traj = trajectory.TrajectoryPropagator(*args, **kwargs) # type: ignore
            for step in range(self._nsteps):
                traj.propagate()
                should_terminate, exit_code = traj.status()
                # FIXME: here
                traj.log_step()
                if should_terminate:
                    traj_lg.log_termination(
                        step=step,
                        exit_code=exit_code
                    )
                    nterminated += 1
                    continue
        lg.log_termination(
            nterminated=nterminated,
            ntraj=self._ntraj,
            prop_duration=self._prop_duration
        )
