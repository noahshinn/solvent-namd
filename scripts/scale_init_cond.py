import torch

from solvent_namd import utils

_INIT_COND_FILE = '../demo/cyp-15-init-cond.pt'
_SAVE_INIT_COND_FILE = 'cyp-15-init-cond.pt'


def main() -> None:
    data = torch.load(_INIT_COND_FILE)
    data = utils.angstrom_to_bohr(data)

    torch.save(data, _SAVE_INIT_COND_FILE)

    print(f'saved to {_SAVE_INIT_COND_FILE}')


if __name__ == '__main__':
    main()
