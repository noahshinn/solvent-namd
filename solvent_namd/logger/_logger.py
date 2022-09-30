"""
STATUS: DEV

"""

import abc
import time


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
        open(f, 'w').close()

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
    def __init__(
            self,
            f: str,
            traj: int,
            ntraj: int,
            delta_t: float,
            nsteps: int,
        ) -> None:
        super().__init__(f, ntraj, delta_t, nsteps)
        self._traj = traj
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
 Step: {step}/{self._nsteps}
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

    def log_step(self) -> None:
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
            ntraj: int,
            prop_duration: float
        ) -> None:
        s = f"""
 -----------------------------------------------------------------------------

 Wall time: {round(time.perf_counter() - self._srt_time, 2)}(s)
 Number of terminated trajectories: {nterminated}
 Number of successful trajectories: {ntraj - nterminated}
 Total number of trajectories: {ntraj}
 Interval: {self._delta_t}(fs)
 Goal propagation duration: {prop_duration}(fs)

 *** Happy Landing ***
 """
        self._log(s)


















