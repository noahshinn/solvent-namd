
# optimizations:
    - replace loop with vmap

"""

import torch

from solvent_namd import utils


def verlet_coords(
        state: int,
        coords: torch.Tensor,
        mass: torch.Tensor,
        velo: torch.Tensor,
        forces: torch.Tensor,
        delta_t: float,
    ) -> torch.Tensor:
    """
    Computes the next atomic coordinate positions using Verlet Integration.

    Args:
        state (int): Electronic energy state of which this molecular system is
            populating.
        coords (torch.Tensor): Coordinate positions of size (N, 3) where N is
            the number of atoms and 3 corresponds to x, y, and z positions in
            Angstroms.
        mass (torch.Tensor): Atomic mass tensor of size (N) where N is the
            number of atoms.
        velo (torch.Tensor): Atomic velocities with respect to the x, y, and z
            axis given by a tensor of size (N, 3) where N is the number of
            atoms.
        forces (torch.Tensor): Atomic forces with respect to the x, y, and z
            axis given by a tensor of size (K, N, 3), where K is the number of
            electronic states and N is the number of atoms.
        delta_t (float): The change in time from the previous snapshot to
            this snapshot in atomic units of time, au.

    Returns:
        next_coords (torch.Tensor): Coordinate positions of size (N, 3) where N is
            the number of atoms and 3 corresponds to x, y, and z positions in
            Angstroms.

    """
    natoms = coords.size(dim=0)
    next_coords = []
    delta_t = utils.fs_to_s(delta_t) / (2.4188843265857 * 1e-17) # type: ignore
    print(delta_t)
    for i in range(natoms):
        delta_pos = (velo[i] * delta_t + 0.5 * forces[state][i] / mass[i] * delta_t ** 2)
        next_coords.extend([coords[i] + delta_pos])
    
    next_coords = torch.stack(next_coords, dim=0)
    # print(next_coords)

    return next_coords


if __name__ == '__main__':
    ntests = 1
    ntests_passed = 0

    state = 2
    coords = torch.rand(51, 3)
    mass = torch.rand(51)
    velo = torch.rand(51, 3)
    forces = torch.rand(3, 51, 3)
    delta_t = 0.05

    next_coords = verlet_coords(
        state=state,
        coords=coords,
        mass=mass,
        velo=velo,
        forces=forces,
        delta_t=delta_t
    )

    assert next_coords.size() == torch.Size([51, 3])
    ntests_passed += 1

    print(f'Passes {ntests_passed}/{ntests} tests!')
