"""
STATUS: DEV

"""

import os
import abc
import time
import glob
import torch

from solvent_namd.utils import bohr_to_angstrom

from typing import List


class Logger:
    __metaclass__ = abc.ABCMeta

    def __init__(
            self,
            root_dir: str,
            ntraj: int,
            delta_t: float,
            nsteps: int
        ) -> None:
        self._root_dir = root_dir 
        self._ntraj = ntraj
        self._delta_t = delta_t
        self._nsteps = nsteps

    @staticmethod
    def _create_dir(d: str) -> None:
        if os.path.exists(d):
            for f in glob.glob(f'{d}/*'):
                os.remove(f)
        else:
            os.makedirs(d)

    @staticmethod
    def _log(msg: str, f: str) -> None:
        with open(f, 'a') as file:
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
            root_dir: str,
            traj: int,
            ntraj: int,
            natoms: int,
            delta_t: float,
            nsteps: int,
            atom_strings: List[str],
            nstates: int
        ) -> None:
        super().__init__(root_dir, ntraj, delta_t, nsteps)
        self._name = f'traj_{traj}'
        self._dir = os.path.join(root_dir, self._name)
        self._create_dir(self._dir)
        self._log_f = os.path.join(self._dir, f'{self._name}.log')
        self._md_xyz_f = os.path.join(self._dir, f'{self._name}.md.xyz')
        self._md_energy_f = os.path.join(self._dir, f'{self._name}.md.energy')
        self._traj = traj
        self._natoms = natoms
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
        self._log(s, self._log_f)
    
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
        self._log(s, self._log_f)

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
        angstroms = bohr_to_angstrom(x)
        assert x.size(dim=1) == 3
        s = ''
        for i, l in enumerate(angstroms):
            a = self._atom_strings[i]
            x_c = l[0]
            y_c = l[1]
            z_c = l[2]
            s += '%-5s%24.16f%24.16f%24.16f\n' % (a, float(x_c), float(y_c), float(z_c))
        return s

    def _log_coords(self, coords: torch.Tensor) -> None:
        formatted = self._format_atomic_info(coords)
        s_log = f"""
  &coordinates in Angstrom
-------------------------------------------------------------------------------
{formatted}-------------------------------------------------------------------------------
"""
        self._log(s_log, self._log_f)
        s_md_xyz = f"""{self._natoms}

{formatted}"""
        self._log(s_md_xyz, self._md_xyz_f)

    def _log_velo(self, velo: torch.Tensor) -> None:
        s = f"""
  &velocities in Angstrom/au
-------------------------------------------------------------------------------
{self._format_atomic_info(velo)}-------------------------------------------------------------------------------
"""
        self._log(s, self._log_f)

    def _log_forces(self, forces: torch.Tensor) -> None:
        s = ''
        for i in range(self._nstates):
            s += f"""
  &gradient state             {i} in Hartree/Angstrom
-------------------------------------------------------------------------------
{self._format_atomic_info(forces[i])}-------------------------------------------------------------------------------
""" 
        self._log(s, self._log_f)

    def _log_energies(self, energies: torch.Tensor) -> None:
        l = energies.size(dim=0)
        s = ''
        for i, e in enumerate(energies):
            s += str(e.item())
            if i != l - 1:
                s += '    '
        self._log(s, self._md_energy_f)
             
    def _log_state(self, state: int) -> None:
        NotImplemented()


class NAMDLogger(Logger):
    def __init__(
            self,
            root_dir: str,
            ntraj: int,
            delta_t: float,
            nsteps: int,
            ncores: int,
            description: str,
            model_name: str,
            natoms: int,
            nstates: int
        ) -> None:
        super().__init__(root_dir, ntraj, delta_t, nsteps)
        self._create_dir(root_dir)
        self._log_f = os.path.join(root_dir, 'namd.log')
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
        self._log(s, self._log_f)
    
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
        self._log(s, self._log_f)

class AdptvSmplLogger(Logger):
    def __init__(
            self,
            root_dir: str,
            ntraj: int,
            delta_t: float,
            nsteps: int,
            ncores: int,
            description: str,
            model_name: str,
            natoms: int,
            nstates: int
        ) -> None:
        super().__init__(root_dir, ntraj, delta_t, nsteps)
        self._create_dir(root_dir)
        self._log_f = os.path.join(root_dir, 'namd.log')
        self._srt_time = time.perf_counter()
        self._ncores = ncores
        self._description = description
        self._model_name = model_name
        self._natoms = natoms
        self._nstates = nstates

    def log_header(self) -> None:
        NotImplemented()

    def log_termination(
            self,
            nstructures: int,
            ntraj: int,
            prop_duration: float
        ) -> None:
        s = f"""
 -----------------------------------------------------------------------------

 Wall time: {time.perf_counter() - self._srt_time:.2f}(s)
 Number of new structures collected: {nstructures}
 Total number of trajectories propagated: {ntraj}
 Interval: {self._delta_t}(fs)
 Goal propagation duration: {prop_duration}(fs)

 *** Happy Landing ***
 """
        self._log(s, self._log_f)

    def log_qm_call(self) -> None:
        NotImplemented()
