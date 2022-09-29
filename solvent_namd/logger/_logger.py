"""
STATUS: DEV

"""


class Logger:
    def __init__(self, f: str) -> None:
        self.f = f 
        open(f, 'w').close()

    def _log(self, msg: str) -> None:
        with open(self.f, 'a') as file:
            file.write(msg)

    def log_header(
            self,
            description: str,
            ncores: int,
            model_name: int,
            natoms: int
        ) -> None:
        s = f"""
 *---------------------------------------------------*
 |                                                   |
 |          Nonadiabatic Molecular Dynamics          |
 |                                                   |
 *---------------------------------------------------*

 Description: {description}
 Number of cores (cpu): {ncores}
 Energy inference model name: {model_name}
 Force inference model name: {model_name}
 Number of atoms: {natoms}

"""
        self._log(s)
    
    def log_happy_ending(
            self,
            wall_time: float,
            traj_id: int,
            ntraj: int,
            prop_duration: float,
            delta_t: float,
            nsteps: int
        ) -> None:
        s = f"""
 -----------------------------------------------------------------------------
 Wall time: {wall_time}(s)
 Trajectory: {traj_id} / {ntraj} 
 Number of steps: {nsteps}
 Interval: {delta_t}(fs)
 Total propagation: {prop_duration}(fs)

 *** Happy Landing ***
"""
        self._log(s)

    def log_termination(
            self,
            wall_time: float,
            traj_id: int,
            ntraj: int,
            delta_t: float,
            step: int,
            nsteps: int,
            exit_code: int
        ) -> None:
        s = f"""
 -----------------------------------------------------------------------------
 Wall time: {wall_time}(s)
 Trajectory: {traj_id} / {ntraj} 
 Step: {step}/{nsteps}
 Interval: {delta_t}(fs)
 Total propagation: {step * delta_t}(fs)
 
 Exit code: {exit_code}

 *** Terminated ***
"""
        self._log(s)
