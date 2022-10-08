"""
STATUS: FINISHED

"""

import yaml
from pathlib import Path

from typing import Dict


def read_yaml(f: str) -> Dict:
    """
    Reads a yml file and returns a Python dictionary.

    Args:
        f (str): yml file

    Returns:
        (Dict): Data dictionary

    """
    assert f.endswith('.yml')

    return yaml.safe_load(Path(f).read_text())['instance']

