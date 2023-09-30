"""

import torch

from typing import List


def write_xyz(
        atom_strings: List[str],
        coords: torch.Tensor,
        file: str
    ) -> None:
    natoms = coords.size(dim=0)
    coords_list = coords.tolist()
    with open(file, 'w') as f:
        f.write(f'{natoms}\n\n')
        for i in range(natoms):
            f.write(f'{atom_strings[i]}\t{coords_list[i][0]}\t{coords_list[i][1]}\t{coords_list[i][2]}')
            if i != natoms - 1:
                f.write('\n')

