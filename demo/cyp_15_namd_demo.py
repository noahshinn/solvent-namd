"""
STATUS: DEV

>> python cyp_15_namd_demo.py cyp-15-input.yml cyp-15-model.pt cyp-15-init-cond.pt cyp-15-one-hot.pt

"""

import sys
import yaml
import torch
from e3nn import o3
from pathlib import Path
from solvent import models

from solvent_namd import NAMD


assert len(sys.argv) == 5
_INPUT_FILE = sys.argv[1]
_MODEL_FILE = sys.argv[2]
_INIT_COND_FILE = sys.argv[3]
_ATOM_TYPES_FILE = sys.argv[4]

model = models.SolventModel(
      irreps_in='3x0e',
      hidden_sizes=[125, 40, 25, 15],
      irreps_out=f'3x0e',
      irreps_node_attr=None,
      irreps_edge_attr=o3.Irreps.spherical_harmonics(3), # type: ignore
      nlayers=3,
      max_radius=4.6,
      nbasis_funcs=8,
      nradial_layers=2,
      nradial_neurons=128,
      navg_neighbors=16.0,
      act=None
)
model.load_state_dict(torch.load(_MODEL_FILE, map_location='cpu')['model'])
model.eval()

init_cond = torch.load(_INIT_COND_FILE)
atom_types = torch.load(_ATOM_TYPES_FILE)

d: dict = yaml.safe_load(Path(_INPUT_FILE).read_text())['instance']

namd = NAMD.deserialize(
        d=d,
        model=model,
        init_cond=init_cond,
        atom_types=atom_types
)
print('load succesful!\n')

namd.run()

print('finished!')
