import torch

from solvent_namd import utils

from typing import Dict, List


def one_hot_to_atom_string(
        one_hot: torch.Tensor,
        string_key: Dict[int, str]
    ) -> List[str]:
    """
    Converts a one-hot atomic encoding to a atom-type string tensor.

    Args:
        one_hot (torch.Tensor): One-hot atomic encoding.
        string_key (Dict): keys: hash value -> values: element abbreviation 

    Returns:
        atom_strings (List(str)): A list of atomic strings.

    """
    atom_strings = [string_key[utils.hash_1d_tensor(atom)] for atom in one_hot]

    return atom_strings
