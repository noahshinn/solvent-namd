"""
STATUS: DEV

"""

import torch
from typing import NamedTuple, List, NewType

PosInt = NewType('PosInt', int)


class XYZData(NamedTuple):
    atom_strings: List[str]
    coords: torch.Tensor 

class TerminationStatus(NamedTuple):
    should_terminate: bool
    exit_code: int

class SurfaceHoppingMetrics(NamedTuple):
    """
    a: state-density matrix
    h: energy matrix
    d: non-adiabatic matrix
    velo: velocities
    hop_type: "NO HOP" | "HOP" | "FRUSTRATED"
    state: current electronic energy state
    log_info: info to log

    """
    a: torch.Tensor
    h: torch.Tensor
    d: torch.Tensor
    velo: torch.Tensor
    hop_type: str 
    state: PosInt
    log_info: str

class P_NACS(NamedTuple):
    p: torch.Tensor 
    nacs: torch.Tensor
