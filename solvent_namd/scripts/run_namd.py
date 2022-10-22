"""
STATUS: DEV

"""

import argparse
from solvent_namd.namd import NAMD
from solvent_namd.utils import Config


def main(args=None) -> None:
    config = _parse_command_line(args)
    namd = NAMD(**config.as_dict()) # type: ignore
    namd.run()

def _parse_command_line(args=None):
    parser = argparse.ArgumentParser(description="Run ML Non-Adiabatic Molecular Dynamics")
    parser.add_argument("config", help="configuration file")
    args = parser.parse_args(args=args)

    config = Config.from_file(args.config) # type: ignore
    return config
