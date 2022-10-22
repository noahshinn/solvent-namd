"""
STATUS: DEV

"""

import torch
import multiprocessing

from typing import Dict

DEFAULT_CONFIGS = {
    'ncores': multiprocessing.cpu_count(),
    'device': 'cpu',
    'init_state': 0, # ground state
    'ntraj': 1,
    'prop_duration': 1000, # in fs
    'delta_t': 0.5, # in fs
    'root': 'results',
    'run_name': 'example-run',
    'model_name': 'Unnamed Model',
    'description': 'No description', # description of run
}


# FIXME: figure out privacy
def _load_model(model_file: str) -> torch.nn.Module:
    model = torch.jit.load(model_file)
    model.eval()
    return model

# TODO: better way of loading initial conditions
def _load_init_cond(init_cond_file: str) -> torch.Tensor:
    d = _load_file(init_cond_file)
    if not 'init_cond' in d:
        raise TypeError('init_cond not in file!')
    return d['init_cond']

def _load_file(filename: str) -> dict:
    f_ext = filename.split('.')[1]
    if f_ext == 'yaml':
        import yaml
        with open(filename) as f:
            return yaml.load(f, Loader=yaml.Loader)
    elif f_ext == 'json':
        import json
        with open(filename) as f:
            return json.load(f)
    elif f_ext == 'pth':
        with open(filename) as f:
            return torch.load(f)
    else:
        raise NotADirectoryError('input file not supported')

class Config:
    """Config class for run parameters."""
    ncores: int
    device: str
    model: torch.nn.Module
    ntraj: int
    prop_duration: float
    delta_t: float
    natoms: int
    nstates: int
    init_cond: torch.Tensor
    init_state: int
    atom_types: torch.Tensor
    one_hot_mass_key: Dict[int, float]
    one_hot_type_key: Dict[int, str]
    root: str
    model_name: str
    description: str

    def __init__(self, d: dict) -> None:
        if d is None:
            self.content = {}
        else:
            self.content = d

        if 'model' not in d:
            raise TypeError('model not specified in input')
        else:
            self.content['model'] = _load_model(d['model'])

        if 'init_cond' not in d:
            raise TypeError('initial conditions file not specified in input')
        else:
            self.content['init_cond'] = _load_init_cond(d['init_cond'])
        self.update()

    def update(self) -> None:
        for k, v in DEFAULT_CONFIGS.items():
            if k not in self.content or self.content[k] is None:
                self.content[k] = v

    def as_dict(self) -> dict:
        return self.content

    @staticmethod
    def from_file(filename: str):
        """Load from file"""
        d = _load_file(filename)
        return Config(**d)
