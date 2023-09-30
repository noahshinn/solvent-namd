from torch_geometric.data.data import Data

from solvent_namd.adptv_smpl import AdaptiveSamplingBase

from typing import List


class AdaptiveSamplingWorker(AdaptiveSamplingBase):
    def __init__(self, dataset: List[Data]) -> None:
        super().__init__(dataset)

    # TODO: handle dict
    def save_structure(self, structure: Data) -> None:
        self._dataset.extend([structure])
