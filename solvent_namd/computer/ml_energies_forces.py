"""
STATUS: DEV

*move unit conversion elsewhere

"""

import torch
from ase import Atoms
from nequip.ase import NequIPCalculator

from solvent_namd.utils import kcal_to_hartree, ev_to_hartree

from typing import Dict, List
from solvent_namd.types import SPEnergiesForces


def _force_grad(
        energies: torch.Tensor,
        pos: torch.Tensor,
        device: str
    ) -> torch.Tensor:
    """
    Computes the gradient of energy with respect to coordinate position
    for every atom for every electronic state for every batch.

    N: number of atoms
    K: number of electronic states

    Args:
        energies (torch.Tensor): Energy tensor of size (K)
        pos (torch.Tensor): Position tensor of size (N, 3)

    Returns:
        jac (torch.Tensor): Jacobian matrix of force tensor of size (K, N, 3)

    """
    nstates = energies.size(dim=0)
    basis_vecs = torch.eye(nstates).to(device=device)
    jac_rows = [torch.autograd.grad(energies, pos, v, retain_graph=True)[0] for v in basis_vecs.unbind()]
    jac = torch.stack(jac_rows, dim=0).neg()
    return jac

# def _ml_forces(energies: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    # nstates = energies.size(dim=0)
    # forces = []
    # for i in range(nstates):
        # f = torch.autograd.grad(
            # -energies[i],
            # pos,
            # create_graph=True,
            # retain_graph=True
        # )[0]
        # forces.append(f * 0.7182231545448303)
    # forces = torch.stack(forces, dim=0)
    # # print(forces)
    # # import sys
    # # sys.exit(0)

    # return forces

# TODO: device
def multi_model_energies_forces(
        structure: Atoms,
        calculators: List[NequIPCalculator],
        units: str = 'kcal'
    ) -> SPEnergiesForces:
    """Computes energies for all energy states."""
    # TODO: better implementation
    energies = []
    forces = []
    for c in calculators:
        c.calculate(atoms=structure, properties=['energy', 'forces'])
        e = c.results['energy']
        f = c.results['forces']
        if units == 'kcal':
            e = kcal_to_hartree(e)
            f = kcal_to_hartree(f)
        elif units == 'ev':
            e = ev_to_hartree(e)
            f = ev_to_hartree(f)
        energies.append(e)
        forces.append(f)
    return SPEnergiesForces(torch.cat(energies, dim=0), torch.cat(forces, dim=0)) 

def single_model_energies_forces(
        model: torch.nn.Module,
        structure: Dict,
        e_shift: float,
        e_scale: float,
        f_scale: float,
        device: str = 'cpu',
        units: str = 'kcal'
    ) -> SPEnergiesForces:
    structure['pos'].requires_grad = True
    y = model(structure)
    dy_dpos = _force_grad(y, structure['pos'], device=device)
    e = y * e_scale + e_shift
    f = dy_dpos * f_scale
    if units == 'kcal':
        e = kcal_to_hartree(e)
        f = kcal_to_hartree(f)
    return SPEnergiesForces(e.detach(), f.detach())
