"""
STATUS: DEV

"""

import os
import glob
import torch
import inspect
from nequip.ase import NequIPCalculator
from joblib import Parallel, delayed
from pathlib import Path

from solvent_namd.utils import (
    InvalidInputError
)
from solvent_namd.computer import (
    one_hot_to_atom_string,
    one_hot_to_mass,
    split_terminated
)
from solvent_namd.trajectory import (
    TrajectoryPropagator
)
from solvent_namd.logger import (
    TrajLogger,
    NAMDLogger,
    # AdptvSmplLogger
)
from solvent_namd.adptv_smpl import AdaptiveSamplingManager

from typing import Dict, Type, TypeVar, List, Optional

T = TypeVar('T', bound='NAMD')


class NAMD():
    """
    Usage:

    >>> import yaml, torch, pathlib
    >>> model = Model(*args, **kwargs)
    >>> model.load_state_dict(torch.load('model.pt'))
    >>> model.eval()
    >>> dataset = ...
    >>> init_cond = torch.load('init_cond.pt')
    >>> atom_types = torch.load('atom_types.pt') 
    >>> constants = yaml.safe_load(pathlib.Path('constants.yml').read_text())['instance']
    >>> namd = NAMD.deserialize(
    ...     d=constants,
    ...     model=model,
    ...     dataset=dataset
    ...     init_cond=init_cond,
    ...     atom_types=atom_types
    ... )
    >>> namd.run()

    """
    root: str
    run_name: str
    description: str
    
    calculators: Dict[str, NequIPCalculator]

    init_cond: torch.Tensor
    init_state: int
    species: List[str]

    ncores: int
    device: str

    ntraj: int
    prop_duration: float
    delta_t: float
    natoms: int
    nstates: int
    nsteps: int
    mass: torch.Tensor

    # TODO: surface hopping
    # _ic_e_thresh: float
    # _isc_e_thresh: float
    # _max_hop: int

    def __init__(
            self,
            root: str,
            run_name: str,
            description: str,
            calculators: Dict[str, NequIPCalculator],
            init_cond: torch.Tensor,
            init_state: int,
            species: List[str],
            ncores: int,
            device: str,
            ntraj: int,
            prop_duration: float,
            delta_t: float,
            natoms: int,
            nstates: int,
            nsteps: int,
            mass: torch.Tensor,
        ) -> None:
        """Manages all trajectory propagations."""
        Path(os.path.join(root, run_name)).mkdir(parents=True, exist_ok=True)

        self._root = root
        self._run_name = run_name
        self._description = description
        self._calculators = calculators

        self._init_cond = init_cond
        self._init_state = init_state
        self._species = species
        self._ncores = ncores
        self._device = device
        self._ntraj = ntraj
        self._prop_duration = prop_duration
        self._delta_t = delta_t
        self._natoms = natoms
        self._nstates = nstates
        self._nsteps = nsteps
        self._mass = mass
    
    def _run_traj(self, traj_num: int) -> int:
        traj_lg = TrajLogger(
            root=self._root,
            run_name=self._run_name,
            traj=traj_num,
            ntraj=self._ntraj,
            natoms=self._natoms,
            delta_t=self._delta_t,
            nsteps=self._nsteps,
            species=self._species,
            nstates=self._nstates
        )
        traj_lg.log_header()
        traj = TrajectoryPropagator(
            device=self._device,
            logger=traj_lg,
            calculators=self._calculators,
            nstates=self._nstates,
            natoms=self._natoms,
            species=self._species,
            mass=self._mass,
            init_state=self._init_state,
            init_coords=self._init_cond[traj_num][0],
            init_velo=self._init_cond[traj_num][1],
            delta_t=self._delta_t,
            nsteps=self._nsteps,
        )
        for step in range(self._nsteps):
            traj.propagate()
            # FIXME: handle adaptive sampling return number of structures
            should_terminate, exit_code = traj.status()
            # FIXME: here
            if should_terminate:
                traj_lg.log_termination(
                    step=step,
                    exit_code=exit_code
                )
                return exit_code
        return 1
    
    def _deploy(self) -> List[int]:
        return Parallel(n_jobs=self._ncores)(delayed(self._run_traj)(i) for i in range(self._ntraj)) # type: ignore

    def run(self) -> None:
        # if self._is_adptv_sampling:
            # lg = AdptvSmplLogger(
                # root_dir=self._log_dir,
                # ntraj=self._ntraj,
                # delta_t=self._delta_t,
                # nsteps=self._nsteps,
                # ncores=self._ncores,
                # description=self._description,
                # model_name=self._model_name,
                # natoms=self._natoms,
                # nstates=self._nstates
            # )
            # # FIXME: count added structures
            # res = self._deploy()
            # lg.log_termination(
                # nstructures=sum(res), # placeholder
                # ntraj=self._ntraj,
                # prop_duration=self._prop_duration
            # ) 
        lg = NAMDLogger(
            root=self._root,
            run_name=self._run_name,
            ntraj=self._ntraj,
            delta_t=self._delta_t,
            nsteps=self._nsteps,
            ncores=self._ncores,
            description=self._description,
            calculators=self._calculators,
            natoms=self._natoms,
            nstates=self._nstates
        )
        res = self._deploy()
        s, t = split_terminated(res)
        lg.log_termination(
            nterminated=t,
            nsuccessful=s,
            prop_duration=self._prop_duration
        )

