import torch
from torch_geometric.data.data import Data

from solvent_namd.adptv_smpl import AdaptiveSamplingBase

from typing import List


class AdaptiveSamplingManager(AdaptiveSamplingBase):
    def __init__(
            self,
            dataset: List[Data]
        ) -> None:
        super().__init__(dataset)

    # TODO: handle dict
    def save_structures(self, structures: List[Data]) -> None:
        self._dataset.extend(structures)

    def retrain(self) -> None:
        NotImplemented() 
