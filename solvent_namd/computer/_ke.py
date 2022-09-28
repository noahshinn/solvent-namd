"""
STATUS: SYNTAX PASS

# optimizations:
    - vmap or dimensional multiplication to avoid for loop

"""

import torch


def ke(mass: torch.Tensor, velo: torch.Tensor) -> torch.Tensor:
    """
    Computes the total kinetic energy of a molecular system.

    Args:
        mass (torch.Tensor): Atomic masses of size (N) where N is the number
            of atoms in the molecular system.
        velo (torch.Tensor): Atomic velocities of size (N, 3) where N is the
            number of atoms in the molecular system and 3 corresponds to
            velocities in the x, y, and z directions.

    Returns:
        ke (torch.Tensor): A scalar value representing the total kinetic
            energy of the molecular system.

    """
    natoms = velo.size(dim=0)
    ke_c = []
    for i in range(natoms):
        ke_c.append(torch.sum(0.5 * mass[i] * velo[i].pow(2)))

    ke = torch.stack(ke_c, dim=0).sum()

    return ke


if __name__ == '__main__':
    ntests = 1
    ntests_passed = 0

    mass = torch.rand(51)
    velo = torch.rand(51, 3)

    ke = kinetic_energy(
        mass=mass,
        velo=velo
    )

    assert ke.size() == torch.Size([])
    ntests_passed += 1

    print(f'Passes {ntests_passed}/{ntests} tests!')
