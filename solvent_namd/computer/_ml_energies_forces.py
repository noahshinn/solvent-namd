"""
STATUS: NOT TESTED

"""

import torch

from solvent_namd import utils

from typing import NamedTuple, Dict


class EnergiesForces(NamedTuple):
    energies: torch.Tensor
    forces: torch.Tensor


def _ml_forces(energies: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    nstates = energies.size(dim=0)
    forces = []
    for i in range(nstates):
        f = torch.autograd.grad(
            -energies[i],
            pos,
            create_graph=True,
            retain_graph=True
        )[0]
        forces.append(f * 0.7182231545448303)
    forces = torch.stack(forces, dim=0)
    # print(forces)
    # import sys
    # sys.exit(0)

    return forces

def ml_energies_forces(
        model: torch.nn.Module,
        structure: Dict,
    ) -> EnergiesForces:
    structure['pos'].requires_grad = True
    e = model(structure) * 0.7182231545448303 + -36152.6796875
    f = _ml_forces(e.squeeze(), structure['pos'])

    return EnergiesForces(utils.ev_to_hartree(e).clone(), utils.ev_to_hartree(f).clone())


if __name__ == '__main__':
    _MODEL_FILE = '../_testing_utils/148.pt'
    _PRELOAD_FILE = '../_testing_utils/_preloaded-1.pkl'
    _NATOM_TYPES = 3
    _HL_SIZE = [125, 40, 25, 15]
    _NUMBER_OF_BASIS = 8
    _RADIAL_LAYERS = 1
    _RADIAL_NEURONS = 128
    _NUM_NEIGHBORS = 16.0
    _NUM_NODES = 20
    _NEIGHBOR_RADIUS = 4.6
    _REDUCE_OUTPUT = False

    ntests = 2
    ntests_passed = 0

    assert ...
    ntests_passed += 1

    print(f'Passes {ntests_passed}/{ntests} tests!')
