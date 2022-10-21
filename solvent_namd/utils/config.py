"""
STATUS: DEV

"""

class Config:
    def __init__(self) -> None:
        NotImplemented()

    @abstractmethod
    def from_file(filename: str):
        """Load from file"""
        f_ext = filename.split('.')[1]
        if f_ext == 'yaml':
            import yaml
            with open(filename) as f:
                d = yaml.load(f, Loader=yaml.Loader)
                return Config(**d)
        elif f_ext == 'json':
            import json
            with open(filename) as f:
                d = json.load(f)
                return Config(**d)
