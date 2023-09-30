import torch


def center_of_pos(coords: torch.Tensor) -> torch.Tensor:
    return coords.mean(dim=0)
