import torch
import decimal

from solvent_namd.types import XYZData
from typing import List


def _lostr_to_lofloat(lst: List[str]) -> List[float]:
    return [float(decimal.Decimal(i)) for i in lst]

def read_xyz(file: str) -> XYZData: 
    atom_strings = []
    coords = []
    with open(file, 'r') as f:
        lines = f.readlines()[2:]
        for l in lines:
            ad = l.split()
            atom_strings.append(ad[0].replace('_', '')) 
            coords.append(torch.FloatTensor(_lostr_to_lofloat(ad[1:])))
        coords_tensor = torch.stack(coords, dim=0)
    return XYZData(atom_strings, coords_tensor)
