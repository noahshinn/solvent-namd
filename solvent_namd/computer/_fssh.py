"""
STATUS: DEV

"""

import math
import torch

from typing import List, Tuple
from solvent_namd import types

_CUTOFF = 1e-16


def _avoid_singularity(
        e_i: torch.Tensor,
        e_j: torch.Tensor,
        state_i: int,
        state_j: int
    ) -> float:
    e_gap = (e_i - e_j).abs().item()
    s = math.copysign(1, state_i - state_j)
    return s * max(e_gap, _CUTOFF)

def _ktdc(
        state_i: int,
        state_j: int,
        cur_energies: torch.Tensor,
        prev_energies: torch.Tensor,
        prev_prev_energies: torch.Tensor,
        delta_t: float
    ) -> float:
    """
    Curvature-driven time-dependent coupling.
    *Truhlar et al J. Chem. Theory Comput. 2022 DOI:10.1021/acs.jctc.1c01080
    
    Args:

    Returns:

    """
    d_vt = _avoid_singularity(
        e_i=cur_energies[state_i],
        e_j=cur_energies[state_j],
        state_i=state_i,
        state_j=state_j
    )
    d_vt_d_t = _avoid_singularity(
        e_i=prev_energies[state_i],
        e_j=prev_energies[state_j],
        state_i=state_i,
        state_j=state_j
    )
    d_v_2d_t = _avoid_singularity(
        e_i=prev_prev_energies[state_i],
        e_j=prev_prev_energies[state_j],
        state_i=state_i,
        state_j=state_j
    )
    d2_v_d_2t = (d_vt - 2 * d_vt_d_t + d_v_2d_t) / delta_t ** 2

    d_ = d2_v_d_2t / d_vt > 0
    if d_ > 0:
        return d_ ** 0.5 / 2
    return 0.0

def _d_p_d_t(
        nstates: int,
        a: torch.Tensor,
        h: torch.Tensor,
        d: torch.Tensor
    ) -> torch.Tensor:
    """ John C. Tully, J. Chem. Phys. 93, 1061 (1990) """
    return torch.stack([a[k, j] * (-1j * h[i, k] - d[i, k])
        - a[i, k] * (-1j * h[k, j] - d[k, j]) for k in range(nstates)
        for j in range(nstates) for i in range(nstates)], dim=0)

def _b_matrix(
        nstates: int,
        a: torch.Tensor,
        h: torch.Tensor,
        d: torch.Tensor
    ) -> torch.Tensor:
    """ John C. Tully, J. Chem. Phys. 93, 1061 (1990) """
    return torch.stack([2 * torch.imag(torch.conj(a[i, j]) * h[i, j]) -2
        * torch.real(torch.conj(a[i, j]) * d[i, j]) for j in range(nstates)
        for i in range(nstates)], dim=0)

def _get_nac(
        natoms: int,
        cur_state: int,
        next_state: int,
        nac_coupling: List[Tuple[int]],
        nacs: torch.Tensor
    ) -> torch.Tensor:
    nac_pair = sorted((cur_state - 1, next_state - 1))
    if nac_pair in nac_coupling and nacs.size(dim=0) > 0:
        nac_pos = nac_coupling.index(nac_pair)
        return nacs[nac_pos]
    return torch.ones(natoms, 3)

def _fssh_log_info() -> None:
    NotImplemented()

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
    ) -> types.SurfaceHoppingMetrics:

    return types.SurfaceHoppingMetrics(a, h, d, velo, hop_type, state, log_info)
