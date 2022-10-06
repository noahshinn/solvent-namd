import os
import torch

from solvent_namd import utils

from typing import Tuple, List

_NTRAJ = 500
_NWATERS = 15
_NAME = 'cyp-s1qd-wigner'
_SAVE_FILE = 'cyp-15-init-cond.pt'
_DIR = '/scratch/adrion.d/photoclick-chemistry/cyp-test/solvent/initcond-gen/wigner-s1'


def closest_n_waters(
        center: torch.Tensor,
        n_waters: int,
        atom_strings: List[str],
        coords: torch.Tensor,
        velo: torch.Tensor
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    natoms = coords.size(dim=0)
    pq = utils.PriorityQueue()
    for i in range(natoms):
        if atom_strings[i] == 'O':
            pq.push(i, utils.eucl_dist(center, coords[i]))
    _coords = list(coords)
    _velo = list(velo)
    new_atom_strings = atom_strings[:6]
    new_coords = _coords[:6]
    new_velo = _velo[:6]
    while n_waters > 0:
        idx = pq.pop()
        for i in range(3):
            new_atom_strings.append(atom_strings[idx + i])
            new_coords.append(_coords[idx + i])
            new_velo.append(_velo[idx + i])
        n_waters -= 1
    return new_atom_strings, torch.stack(new_coords, dim=0), torch.stack(new_velo, dim=0)

def main() -> None:
    data = []
    for i in range(1, _NTRAJ + 1):
        dir_ = os.path.join(_DIR, f'{_NAME}-{i}')
        xyz_file = os.path.join(dir_, f'{_NAME}-{i}.xyz')
        velo_file = os.path.join(dir_, f'{_NAME}-{i}.velocity.xyz')
        atom_strings, coords = utils.read_xyz(xyz_file)
        coords = utils.center_coords(coords)
        center = utils.center_of_pos(coords)
        velo = utils.read_velo(velo_file)
        new_atom_strings, new_coords, new_velo = closest_n_waters(
            center=center,
            n_waters=_NWATERS,
            atom_strings=atom_strings,
            coords=coords,
            velo=velo
        )
        if i == 245:
            utils.write_xyz(new_atom_strings, new_coords, 'sample.xyz')

        traj_data = torch.stack((new_coords, new_velo), dim=0)
        data.append(traj_data) 
        print(xyz_file)
    
    save_data = torch.stack(data, dim=0)
    torch.save(save_data, _SAVE_FILE)
    print('finished:', save_data.shape)
    print(f'saved to {_SAVE_FILE}')
    print(f'sample saved to sample.xyz')


if __name__ == '__main__':
    main()
