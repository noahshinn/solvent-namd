"""
STATUS: DEV

"""

import abc
from torch_geometric.data.data import Data

from typing import List


class AdaptiveSamplingBase:
    __metaclass__ = abc.ABCMeta

    def __init__(self, dataset: List[Data]) -> None:
        self._dataset = dataset

    def get_dataset(self) -> List[Data]:
        return self._dataset
