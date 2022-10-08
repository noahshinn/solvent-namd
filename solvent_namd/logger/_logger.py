"""
STATUS: DEV

"""

import os
import abc
import time
import torch

from typing import List


class Logger:
    __metaclass__ = abc.ABCMeta

    def __init__(
            self,
            f: str,
            ntraj: int,
            delta_t: float,
            nsteps: int
        ) -> None:
        self._f = f 
        self._ntraj = ntraj
        self._delta_t = delta_t
        self._nsteps = nsteps
        if os.path.exists(f):
            open(f, 'w').close()
        else:
            print(f'creating {f}')
            open(f, 'x').close()

    def _log(self, msg: str) -> None:
        with open(self._f, 'a') as file:
            file.write(msg)

    @abc.abstractmethod
    def log_header(self) -> None:
        """ Abstract method """
        return

    @abc.abstractmethod
    def log_termination(self) -> None:
        """ Abstract method """
        return


class TrajLogger(Logger):
    _atom_strings: List[str]

    def __init__(
            self,
            f: str,
            traj: int,
            ntraj: int,
            delta_t: float,
            nsteps: int,
            atom_strings: List[str],
            nstates: int
        ) -> None:
        super().__init__(f, ntraj, delta_t, nsteps)
        self._traj = traj
        self._atom_strings = atom_strings 
        self._nstates = nstates
        self._srt_time = time.perf_counter()

    def log_header(self) -> None:
        s = f"""
 *---------------------------------------------------*
 |                                                   |
 |          Nonadiabatic Molecular Dynamics          |
 |                                                   |
 *---------------------------------------------------*

 Trajectory ID: traj-{self._traj}

"""
        self._log(s)
    
    def log_termination(
            self,
            step: int,
            exit_code: int,
        ) -> None:
        s = f"""
 -----------------------------------------------------------------------------

 Wall time: {round(time.perf_counter() - self._srt_time, 2)}(s)
 Trajectory: {self._traj} / {self._ntraj - 1} 
 Steps: {step + 1}/{self._nsteps}
 Interval: {self._delta_t}(fs)
 Total propagation: {step * self._delta_t}(fs)
 """
        if exit_code == 0:
            s += f"""

 *** Happy Landing ***
"""
        else:
            s += f"""
 Exit code: {exit_code}

 *** Terminated ***
"""
        self._log(s)

    def log_step(
            self,
            coords: torch.Tensor,
            velo: torch.Tensor,
            forces: torch.Tensor,
            energies: torch.Tensor,
            state: int
        ) -> None:
        self._log_coords(coords)
        self._log_velo(velo)
        self._log_forces(forces)
        # self._log_energies(energies)
        # self._log_state(state)

    def _format_atomic_info(self, x: torch.Tensor) -> str:
        assert x.size(dim=1) == 3
        s = ''
        for i, l in enumerate(x):
            a = self._atom_strings[i]
            x_c = l[0]
            y_c = l[1]
            z_c = l[2]
            s += '%-5s%24.16f%24.16f%24.16f\n' % (a, float(x_c), float(y_c), float(z_c))
        return s

    def _log_coords(self, coords: torch.Tensor) -> None:
        s = f"""
  &coordinates in Angstrom
-------------------------------------------------------------------------------
{self._format_atomic_info(coords)}-------------------------------------------------------------------------------
"""
        self._log(s)

    def _log_velo(self, velo: torch.Tensor) -> None:
        s = f"""
  &velocities in Angstrom/au
-------------------------------------------------------------------------------
{self._format_atomic_info(velo)}-------------------------------------------------------------------------------
"""
        self._log(s)

    def _log_forces(self, forces: torch.Tensor) -> None:
        s = ''
        for i in range(self._nstates):
            s += f"""
  &gradient state             {i} in Hartree/Angstrom
-------------------------------------------------------------------------------
{self._format_atomic_info(forces[i])}-------------------------------------------------------------------------------
""" 
        self._log(s)

    def _log_energies(self, energies: torch.Tensor) -> None:
        NotImplemented()
             
    def _log_state(self, state: int) -> None:
        NotImplemented()


class NAMDLogger(Logger):
    def __init__(
            self,
            f: str,
            ntraj: int,
            delta_t: float,
            nsteps: int,
            ncores: int,
            description: str,
            model_name: str,
            natoms: int,
            nstates: int
        ) -> None:
        super().__init__(f, ntraj, delta_t, nsteps)
        self._srt_time = time.perf_counter()
        self._ncores = ncores
        self._description = description
        self._model_name = model_name
        self._natoms = natoms
        self._nstates = nstates

    def log_header(self) -> None:
        s = f"""
 *---------------------------------------------------*
 |                                                   |
 |          Nonadiabatic Molecular Dynamics          |
 |                                                   |
 *---------------------------------------------------*

 Description: {self._description}
 Number of cores (cpu): {self._ncores}
 Energy inference model name: {self._model_name}
 Force inference model name: {self._model_name}
 Number of atoms: {self._natoms}
 Number of electronic states: {self._nstates}

"""
        self._log(s)
    
    def log_termination(
            self,
            nterminated: int,
            nsuccessful: int,
            prop_duration: float
        ) -> None:
        s = f"""
 -----------------------------------------------------------------------------

 Wall time: {time.perf_counter() - self._srt_time:.2f}(s)
 Number of terminated trajectories: {nterminated}
 Number of successful trajectories: {nsuccessful}
 Total number of trajectories: {nsuccessful + nterminated}
 Interval: {self._delta_t}(fs)
 Goal propagation duration: {prop_duration}(fs)

 *** Happy Landing ***
 """
        self._log(s)
