"""
STATUS: DEV

"""

import torch

from typing import List, Dict
from solvent_namd.types import SPEnergiesForces


class QMComputer:
    def __init__(self) -> None:
        self._job_hash: Dict[int, str] = {}
        self._next_job_id = 0

    def _job_id_to_file(self, job_id: int) -> str:
        if job_id in self._job_hash:
            return self._job_hash[job_id]
        # TODO: throw job not submitted error 
        NotImplemented()
        return '' # placeholder

    def _has_completed(self, job_id: int) -> bool:
        job_path = self._job_id_to_file(job_id=job_id)    
        with open(job_path, 'r') as f:
            return '# Happy landing! #' in f.read()

    def _read_e(self, job_id: int) -> torch.Tensor:
        NotImplemented()
        # TODO: read file to get energies
        return torch.zeros(1) # placeholder
    
    def _read_f(self, job_id: int) -> torch.Tensor:
        NotImplemented()
        # TODO: read file to get forces 
        return torch.zeros(1) # placeholder
    
    def _gen_inp_file(self) -> None:
        NotImplemented()
    
    def _gen_slurm_file(self) -> None:
        NotImplemented()

    def _gen_out_file(self) -> None:
        NotImplemented()
    
    def _send(self, job_id: int) -> None:
        # TODO: run bash command
        # get path of output file
        out_file = '' # placeholder
        self._job_hash[self._next_job_id] = out_file
        self._next_job_id += 1

    def sp(self, species: List[str], coords: torch.Tensor) -> SPEnergiesForces:
        # submit job
        
        # async wait for sp to finish
        job_id = 0 # placeholder
        if self._has_completed(job_id=job_id):
            e = self._read_e(job_id) 
            f = self._read_f(job_id)
            return SPEnergiesForces(e, f)
        # throw error
