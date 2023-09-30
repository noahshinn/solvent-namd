"""

import torch


def read_velo(f: str) -> torch.Tensor:
    with open(f, 'r') as file:
        data = file.readlines()
        velo = []
        for l in data:
            l_data = [float(i) for i in l.split()]
            velo.append(torch.FloatTensor(l_data))
    return torch.stack(velo, dim=0)
