"""
STATUS: DEV

Zhu-Nakamura Theory for generalized surface hopping.
J. Chem. Phys. 102, 7448 (1995); https://doi.org/10.1063/1.469057

Formulas given in supporting information below:
http://www.rsc.org/suppdata/c8/cp/c8cp02651c/c8cp02651c1.pdf

# HERE
    - fix named tuple type error
    - intersystem_crossing call
    - check if is valid surface hop
    - reflect, adjust velocities

"""

import torch

from solvent_namd.computer import (
    internal_conversion,
    intersystem_crossing,
    # is_valid_surface_hop,
    # adjust_velo_after_hop
)

from typing import NamedTuple


class SurfaceHoppingMetrics(NamedTuple):
    """
    a: state-density matrix
    h: energy matrix
    d: non-adiabatic matrix
    velo: velocities
    hop_type: "NO HOP" | "HOP" | "FRUSTRATED"
    state: current electronic energy state

    """
    a: torch.Tensor
    h: torch.Tensor
    d: torch.Tensor
    velo: torch.Tensor
    hop_type: str 
    state: int


def sh(
        state: int,
        state_mult: torch.Tensor,
        mass: torch.Tensor,
        coord: torch.Tensor,
        coord_prev: torch.Tensor,
        coord_prev_prev: torch.Tensor,
        velo: torch.Tensor,
        energies: torch.Tensor,
        energies_prev: torch.Tensor,
        energies_prev_prev: torch.Tensor,
        forces: torch.Tensor,
        forces_prev: torch.Tensor,
        forces_prev_prev: torch.Tensor,
        ke: torch.Tensor,
        ic_e_thresh: float,
        isc_e_thresh: float,
        max_hop: int
    ) -> SurfaceHoppingMetrics:
    """
    ...

    Args:

    Returns:

    """
    nstates = energies.size(dim=0)
    new_state = state
    v = velo
    g_c = torch.zeros(nstates)  # hopping probabilties
    g = 0.0  # acc hopping probability
    target_mult = state_mult[state]
    state_idxs = torch.argsort(energies)
    z = torch.rand(1)

    for i in range(nstates):
        if i == state:
            continue
        
        state_mult = state_mult[i]

        if state_mult == target_mult:
            p, nacs = internal_conversion( # type: ignore
                cur_state=state,
                other_state=i,
                mass=mass,
                coord=coord,
                coord_prev=coord_prev,
                coord_prev_prev=coord_prev_prev,
                velo=velo,
                energies=energies,
                energies_prev=energies_prev,
                energies_prev_prev=energies_prev_prev,
                forces=forces,
                forces_prev=forces_prev,
                forces_prev_prev=forces_prev_prev,
                ke=ke,
                ic_e_thresh=ic_e_thresh
            )
        else:
            p = intersystem_crossing(*args, **kwargs) # type: ignore
        g_c[i] += p
    
    for i in range(nstates):
        g += g_c[state_idxs[i]]
        nhop = torch.abs(state_idxs[i] - state)
        if g > z and 0 < nhop <= max_hop:
            if is_valid_surface_hop(*args, **kwargs): # type: ignore
                new_state = state_idxs[i]
                v = adjust_velo_after_hop(*args, **kwargs) # type: ignore
            else:
                v = velo
    
    return SurfaceHoppingMetrics(a, h, d, v, hop_type, new_state)
