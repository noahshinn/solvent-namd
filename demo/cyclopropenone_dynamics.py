"""
STATUS: DEV

"""

import sys
import yaml
from pathlib import Path

from solvent_namd import NAMD


# input file
assert len(sys.argv) == 2
_INPUT_FILE = sys.argv[1]

# load yaml file to python dict
constants: dict = yaml.safe_load(Path(_INPUT_FILE).read_text())['instance']

namd = NAMD.deserialize(constants)
namd.run()

print('finished!')
