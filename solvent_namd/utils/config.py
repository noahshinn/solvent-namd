"""
STATUS: DEV

"""

import torch
import multiprocessing
from nequip.ase import NequIPCalculator

from solvent_namd.computer import get_mass

from typing import Dict, List, Union

SUPPORTED_CONFIGS = {
    'root': 'results',
    'run_name': 'example-run',
    'description': 'No description', # description of run

    'models': {'s0': 's0.pth'},

    'init_cond': 'init_cond.pth',
    'init_state': 0, # ground state
    'species': '', # atom type

    'ncores': multiprocessing.cpu_count(),
    'device': 'cpu',
    'ntraj': 100,
    'prop_duration': 1000, # in fs
    'delta_t': 0.5, # in fs
}


# TODO: better way of loading initial conditions
def _load_init_cond(init_cond_file: str) -> torch.Tensor:
    d = _load_file(init_cond_file)
    if not 'init_cond' in d:
        raise TypeError('init_cond not in file!')
    assert d['init_cond'].size(dim=0) > 0
    return d['init_cond']

def _load_file(filename: str) -> dict:
    """Loads a supported file type"""
    f_ext = filename.split('.')[1]
    if f_ext == 'yaml':
        import yaml
        with open(filename) as f:
            return yaml.load(f, Loader=yaml.Loader)
    elif f_ext == 'json':
        import json
        with open(filename) as f:
            return json.load(f)
    elif f_ext == 'pth' or f_ext == 'pt':
        with open(filename) as f:
            return torch.load(f)
    else:
        raise NotADirectoryError('input file not supported')

class Config:
    """Config class for run parameters."""
    root: str
    run_name: str
    description: str
    
    calculators: Dict[str, NequIPCalculator]

    init_cond: torch.Tensor
    init_state: int
    species: List[str]

    ncores: int
    device: str

    ntraj: int
    prop_duration: float
    delta_t: float
    natoms: int
    nstates: int
    nsteps: int
    mass: torch.Tensor

    def __init__(
            self,
            root: str = SUPPORTED_CONFIGS['root'],
            run_name: str = SUPPORTED_CONFIGS['run_name'],
            description: str = SUPPORTED_CONFIGS['description'],
            models: Dict[str, str] = SUPPORTED_CONFIGS['models'],
            init_cond: str = SUPPORTED_CONFIGS['init_cond'] ,
            init_state: int = SUPPORTED_CONFIGS['init_state'] ,
            species: Union[str, List[str]] = SUPPORTED_CONFIGS['species'],
            device: str = SUPPORTED_CONFIGS['device'],
            ntraj: int = SUPPORTED_CONFIGS['ntraj'],
            prop_duration: int = SUPPORTED_CONFIGS['prop_duration'],
            delta_t: float = SUPPORTED_CONFIGS['delta_t'],
        ) -> None:
        # set logging info
        self.root = root
        self.run_name = run_name
        self.description = description

        # load calculators
        self.calculators = {}
        for k, v in models:
            self.calculators[k] = NequIPCalculator.from_deployed_model(
                model_path=v,
                device=device,
            )

        # load initial conditions
        self.init_cond = _load_init_cond(init_cond)
        self.init_state = init_state
        if isinstance(species, list):
            self.species = species
        elif isinstance(species, str):
            self.species = species.split() 
        else:
            raise TypeError(f'type `{type(species)}` not supported')
        assert self.natoms == len(self.species)

        # set dynamics info
        self.ntraj = ntraj
        self.prop_duration = prop_duration
        self.delta_t = delta_t
        self.natoms = self.init_cond.size(dim=2)
        self.nstates = len(self.calculators.keys()) > 0
        assert self.nstates > 0
        self.nsteps = int(prop_duration / delta_t)
        self.mass = get_mass(self.species)

    def as_dict(self) -> dict:
        return self.__dict__

    @staticmethod
    def from_file(filename: str):
        """Load from file"""
        d = _load_file(filename)
        return Config(**d)
