"""
STATUS: DEV

Internal conversion probability by the Zhu-Nakamura Theory.

Formula found in the source below:
http://www.rsc.org/suppdata/c8/cp/c8cp02651c/c8cp02651c1.pdf

"""

import math
import torch


def _delta_e(
        cur_state: int,
        other_state: int,
        energies: torch.Tensor,
        energies_prev: torch.Tensor,
        energies_prev_prev: torch.Tensor
    ) -> torch.Tensor:
    """
    Computes an energy gap tensor of size (3) between the current
    state and the state in question.
        - [0] current energy gap
        - [1] prev energy gap
        - [2] prev prev energy gap

    """
    d_E = [
        energies[other_state] - energies[cur_state],
        energies_prev[other_state] - energies_prev[cur_state],
        energies_prev_prev[other_state] - energies_prev_prev[cur_state]
    ]
    return torch.stack(d_E, dim=0).abs()

def _gsh_internal_conversion(
        cur_state: int,
        other_state: int,
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
        ic_e_thresh: float
    ) -> P_NACS:
    """
    Computes a tensor of internal conversion hopping probabilities between
    the current electronic state and every other state.

    Args:
        cur_state (int): Electronic energy state of which this molecular system is
            populating.
        other_state (int): Electronic energy state of the other electronic
            state in question
        mass (torch.Tensor): Atomic mass tensor of size (N) where N is the
            number of atoms.
        coords (torch.Tensor): Coordinate positions of size (N, 3) where N is
            the number of atoms and 3 corresponds to x, y, and z positions in
            Angstroms.
            *current coordinates
        coords_prev (torch.Tensor): Coordinate positions of size (N, 3) where
            N is the number of atoms and 3 corresponds to x, y, and z positions
            in Angstroms.
            *prev coordinates
        coords_prev_prev (torch.Tensor): Coordinate positions of size (N, 3)
            where N is the number of atoms and 3 corresponds to x, y, and z
            positions in Angstroms.
            *prev prev coordinates
        velo (torch.Tensor): Atomic velocities with respect to the x, y, and z
            axis given by a tensor of size (N, 3) where N is the number of
            atoms.
        energies (torch.Tensor): A potential energy tensor of size K where K
            is the number of electronic states.
            *current energies
        energies_prev (torch.Tensor): A potential energy tensor of size K where
            K is the number of electronic states.
            *prev step energies
        energies_prev_prev (torch.Tensor): A potential energy tensor of size K
            where K is the number of electronic states.
            *prev prev step energies
        forces (torch.Tensor): An atomic force tensor with respect to the x, y,
            and z axis given by a tensor of size (K, N, 3), where K is the
            number of electronic states and N is the number of atoms.
            *current forces
        forces_prev (torch.Tensor): An atomic force tensor with respect to the
            x, y, and z axis given by a tensor of size (K, N, 3), where K is
            the number of electronic states and N is the number of atoms.
            *prev step forces
        forces_prev_prev (torch.Tensor): An atomic force tensor with respect to
            the x, y, and z axis given by a tensor of size (K, N, 3), where K
            is the number of electronic states and N is the number of atoms.
            *prev prev step forces
        ke (torch.Tensor): A scalar value representing the total kinetic
            energy of the molecular system.
        ic_e_thresh (torch.Tensor): energy gap threshold to compute Zhu-
            Nakamura surface hopping between the same spin states
         
    Returns:
        p, nacs (torch.Tensor, torch.Tensor): The hopping probability between
            the two given electronic states and the non-adiabatic matrix.

    """

    delta_e = _delta_e(cur_state, other_state, energies, energies_prev, energies_prev_prev)
    e = energies_prev[cur_state] + ke
    avg_e = (energies_prev[other_state] + energies_prev[cur_state]) / 2

    # check hop condition

    low_state = min(other_state, cur_state)
    high_state = max(other_state, cur_state)

    bt = -1 / (coord - coord_prev_prev) # (51, 3)
    tf1_1 = forces[low_state] * (coord_prev - coord_prev_prev) # (51, 3)
    tf2_1 = forces_prev_prev[high_state] * (coord_prev - coord) # (51, 3)
    f_ia_1 = bt * (tf1_1 - tf2_1) # (51, 3)

    tf1_2 = forces[high_state] * (coord_prev - coord_prev_prev) # (51, 3)
    tf2_2 = forces_prev_prev[low_state] * (coord_prev - coord) # (51, 3)
    f_ia_2 = bt * (tf1_2 - tf2_2) # (51, 3)

    f_a = torch.sum((f_ia_2 - f_ia_1) ** 2 / mass) ** 0.5
    f_b = torch.sum(f_ia_1 * f_ia_2 / mass).abs() ** 0.5
    a_2 = (f_a * f_b) / (2 * delta_e ** 3)
    n = math.pi / (4 * a_2 ** 0.5)

    b_2 = (e - avg_e) * f_a / (f_b * delta_e)
    s = torch.sum(f_ia_1 * f_ia_2).sign()
    m = 2 / ((b_2 + torch.abs(b_2 ** 2 + s)) ** 0.5)

    p = torch.exp(-n * m)

    pnacs = (f_ia_2 - f_ia_1) / mass.pow(2)
    nacs = pnacs / torch.sum(pnacs ** 2).pow(0.5)
    
    return types.P_NACS(p, nacs)


if __name__ == '__main__':
    ntests = 1
    ntests_passed = 0

    _NATOMS = 51
    _NSTATES = 3

    cur_state = 0
    other_state = 1
    mass = torch.rand(_NATOMS)
    coord = torch.rand(_NATOMS, 3)
    coord_prev = torch.rand(_NATOMS, 3)
    coord_prev_prev = torch.rand(_NATOMS, 3)
    velo = torch.rand(_NATOMS, 3)
    energies = torch.rand(_NSTATES)
    energies_prev = torch.rand(_NSTATES)
    energies_prev_prev = torch.rand(_NSTATES)
    forces = torch.rand(_NSTATES, _NATOMS, 3)
    forces_prev = torch.rand(_NSTATES, _NATOMS, 3)
    forces_prev_prev = torch.rand(_NSTATES, _NATOMS, 3)
    ke = torch.rand(1)
    ic_e_thresh = 0.3 

    d_e = _delta_e(
        cur_state=cur_state,
        other_state=other_state,
        energies=energies,
        energies_prev=energies_prev,
        energies_prev_prev=energies_prev_prev
    )

    assert d_e.size() == torch.Size([3])
    ntests_passed += 1

    p, nacs = gsh_internal_conversion(
        cur_state=cur_state,
        other_state=other_state,
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

    print('p:', p)
    print('nacs:', nacs)
    # assert p.size == torch.Size([]) and nacs.size() == torch.Size([])
    ntests_passed += 1

    print(f'Passes {ntests_passed}/{ntests} tests!')
