"""
STATUS: DEV

"""

import os
import glob
import torch
import inspect

from solvent_namd import utils, logger, trajectory, computer

from typing import Dict, Type, TypeVar, List

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
    _init_state: int
    _atom_types: torch.Tensor
    _mass: torch.Tensor
    _atom_strings: List[str]
    _ic_e_thresh: float
    _isc_e_thresh: float
    _max_hop: int
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
            init_state: int,
            atom_types: torch.Tensor,
            one_hot_mass_key: Dict[int, float],
            one_hot_type_key: Dict[int, str],
            ic_e_thresh: float,
            isc_e_thresh: float,
            max_hop: int,
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
        assert init_cond.shape == torch.Size([ntraj, 2, natoms, 3])

        self._ncores = ncores
        self._model = model
        self._ntraj = ntraj
        self._prop_duration = prop_duration
        self._delta_t = delta_t
        self._natoms = natoms
        self._nstates = nstates 
        self._init_cond = init_cond
        self._atom_types = atom_types
        self._mass = computer.one_hot_to_mass(atom_types, one_hot_mass_key)
        self._atom_strings = computer.one_hot_to_atom_string(atom_types, one_hot_type_key)
        self._init_state = init_state
        self._nsteps = int(prop_duration / delta_t)
        self._ic_e_thresh = ic_e_thresh
        self._isc_e_thresh = isc_e_thresh 
        self._max_hop = max_hop
        self._log_dir = log_dir
        if os.path.exists(log_dir):
            for f in glob.glob(f'{log_dir}/*'):
                os.remove(f)
        else:
            os.makedirs(log_dir)
        self._model_name = model_name
        self._description = description

    @classmethod
    def deserialize(
            cls: Type[T],
            d: Dict,
            model: torch.nn.Module,
            init_cond: torch.Tensor,
            atom_types: torch.Tensor
        ) -> T:
        d['model'] = model
        d['init_cond'] = init_cond
        d['atom_types'] = atom_types
        params = inspect.getfullargspec(NAMD.__init__).args
        for k in d:
            if not k in params:
                raise utils.InvalidInputError(f'input `{k}` was given but is not a valid input')
        for p in params:
            if p != 'self' and not p in d:
                raise utils.InvalidInputError(f'input `{p}` was not given in input')
        return cls(**d)

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
                atom_strings=self._atom_strings,
                nstates=self._nstates
            )
            traj_lg.log_header()
            traj = trajectory.TrajectoryPropagator(
                logger=traj_lg,
                model=self._model,
                init_state=self._init_state,
                nstates=self._nstates,
                natoms=self._natoms,
                atom_types=self._atom_types,
                mass=self._mass,
                atom_strings=self._atom_strings,
                init_coords=self._init_cond[i][0],
                init_velo=self._init_cond[i][1],
                delta_t=self._delta_t,
                nsteps=self._nsteps,
                ic_e_thresh=self._ic_e_thresh,
                isc_e_thresh=self._isc_e_thresh,
                max_hop=self._max_hop
            )
            for step in range(self._nsteps):
                traj.propagate()
                should_terminate, exit_code = traj.status()
                # FIXME: here
                if should_terminate:
                    traj_lg.log_termination(
                        step=step,
                        exit_code=exit_code
                    )
                    nterminated += 1
                    break 
        lg.log_termination(
            nterminated=nterminated,
            ntraj=self._ntraj,
            prop_duration=self._prop_duration
        )
