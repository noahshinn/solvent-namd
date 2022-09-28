"""
STATUS: DEV

"""

import sys

from solvent_namd import NAMD

assert len(sys.argv) == 2
_INPUT_FILE = sys.argv[1]

**constants = rd_input(_INPUT_FILE) # type: ignore


def main() -> None:
    namd = NAMD(**constants) # type: ignore
    namd.run()


if __name__ == '__main__':
    main()
