"""
STATUS: DEV

"""

import torch
import multiprocessing

from typing import Dict, Optional

SUPPORTED_CONFIGS = {
    'ncores': multiprocessing.cpu_count(),
    'device': 'cpu',
    'model': 'best_model.pth',
    'init_cond': 'init_cond.pth', # initial conditions file
    'init_state': 0, # ground state
    'ntraj': 1,
    'prop_duration': 1000, # in fs
    'delta_t': 0.5, # in fs
    'root': 'results',
    'run_name': 'example-run',
    'model_name': 'Unnamed Model',
    'description': 'No description', # description of run
}


def _set_values(d: dict) -> dict:
    for k, v in SUPPORTED_CONFIGS.items():
        if k not in d or d[k] is None:
            d[k] = v
    return d

def _load_model(model_file: str) -> torch.nn.Module: # type: ignore
    NotImplemented()

def _load_init_cond(init_cond_file: str) -> torch.Tensor: # type: ignore
    NotImplemented()

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

    def __init__(self, d: Optional[dict] = None) -> None:
        if d is None:
            self.content = {}
        else:
            self.content = d

    def update(self, d: dict) -> None:
        for k, v in d.items():
            self.content[k] = v

    @staticmethod
    def from_file(filename: str):
        """Load from file"""
        f_ext = filename.split('.')[1]
        if f_ext == 'yaml':
            import yaml
            with open(filename) as f:
                d = yaml.load(f, Loader=yaml.Loader)
                return Config(**_set_values(d))
        elif f_ext == 'json':
            import json
            with open(filename) as f:
                d = json.load(f)
                return Config(**_set_values(d))

    @staticmethod
    def from_dict(d: dict):
        c = Config()
        c.update(d)
