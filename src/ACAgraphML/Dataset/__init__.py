from torch_geometric.datasets import ZINC
from torch_geometric.transforms import BaseTransform
from typing import Literal
from enum import Enum
from .BONDataset import BONDataset


class ZINC_Dataset(Enum):
    def loadDatasetZINC(split: Literal['train', 'val', 'test'] = 'train', subset: bool = False, transform: BaseTransform = None):
        """
        Load the ZINC dataset.
        Args:
            split (str): The dataset split to load. Options are 'train', 'val', or 'test'.
            subset (bool): If True, load a smaller subset of the dataset.
        Returns:
            dataset (ZINC): The loaded ZINC dataset.
        """
        return ZINC(
            root='./data/ZINC',
            split=split,
            subset=subset,
            transform=transform
        )

    def __init__(self, split: Literal['train', 'val', 'test'] = 'train', subset: bool = False):
        super().__init__()
        self.split = split
        self.subset = subset

    def load(self, transform: BaseTransform = None):
        """
        Load the ZINC dataset based on the specified split and subset.
        Returns:
            dataset (ZINC): The loaded ZINC dataset.
        """
        return ZINC_Dataset.loadDatasetZINC(split=self.split, subset=self.subset, transform=transform)

    FULL_TRAIN = 'train', False
    FULL_VAL = 'val', False
    FULL_TEST = 'test', False
    SMALL_TRAIN = 'train', True
    SMALL_VAL = 'val', True
    SMALL_TEST = 'test', True
