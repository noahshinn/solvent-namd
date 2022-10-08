"""
STATUS: DEV

"""

import torch

from typing import List, Tuple


def fssh(
        prev_a: torch.Tensor,
        prev_h: torch.Tensor,
        prev_d: torch.Tensor,
        nacs: torch.Tensor,
        socs: torch.Tensor,
        nsubsteps: int,
        step_size: float,
        iteration: int,
        nstates: int,
        cur_state: int,
        nmax_hop: int,
        e_deco: float,
        adjust: int,         # FIXME: better name and type
        reflect: int,        # FIXME: better name and type
        nacs_type: str,
        mass: torch.Tensor,
        cur_velo: torch.Tensor,
        cur_energies: torch.Tensor,
        prev_energies: torch.Tensor,
        prev_prev_energies: torch.Tensor,
        ke: torch.Tensor,
        nac_coupling: List[Tuple[int]],
        soc_coupling: List[Tuple[int]],
        state_mult: List[int]
    ) -> None:
    NotImplemented()
