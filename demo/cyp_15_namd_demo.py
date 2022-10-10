"""
STATUS: DEV

>> python cyp_15_namd_demo.py cyp-15-input.yml cyp-15-model.pt cyp-15-init-cond.pt cyp-15-one-hot.pt

"""

import sys
import yaml
import torch
from pathlib import Path
from solvent.models import Model

from solvent_namd import NAMD


INPUT_FILE = './cyp-15-input.yml'
INIT_COND_FILE = './cyp-15-init-cond.pt'
ATOM_TYPES_FILE = './cyp-15-one-hot.pt'
CHKPT_FILE = './test_params.pt'
NATOM_TYPES = 3
NSTATES = 3

model = Model(
    irreps_in=f'{NATOM_TYPES}x0e',
    hidden_sizes=[125, 40, 25, 15],
    irreps_out=f'{NSTATES}x0e',
    nlayers=4,
    max_radius=4.6,
    nbasis_funcs=8,
    nradial_layers=3,
    nradial_neurons=128,
    navg_neighbors=16.0,
    cache=None
)
model.load_state_dict(torch.load(CHKPT_FILE, map_location='cpu')['model'])
model.eval()
print('model loaded')

init_cond = torch.load(INIT_COND_FILE)
atom_types = torch.load(ATOM_TYPES_FILE)
print('initial conditions loaded')

d: dict = yaml.safe_load(Path(INPUT_FILE).read_text())['instance']

namd = NAMD.deserialize(
        d=d,
        model=model,
        init_cond=init_cond,
        atom_types=atom_types
)
print('NAMD object initialized\n')

print('running dynamics!')
namd.run()
print('finished!')
