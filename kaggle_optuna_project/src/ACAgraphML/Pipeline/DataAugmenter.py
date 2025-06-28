import torch_geometric.transforms as T
from torch_geometric.data import Dataset
from tqdm.notebook import tqdm


class DataAugmenter:
    """
    A utility class for applying data augmentation transforms to PyTorch Geometric datasets.

    This class takes a dataset and a list of transforms, composes them together,
    and applies them to all data samples in the dataset to create an augmented version.

    Attributes:
        dataset (Dataset): The original PyTorch Geometric dataset to augment.
        transform (T.Compose): The composed transformation pipeline.
    """

    def __init__(self, dataset: Dataset, transforms: list[T.BaseTransform]):
        """
        Initialize the DataAugmenter with a dataset and list of transforms.

        Args:
            dataset (Dataset): A PyTorch Geometric dataset containing graph data.
            transforms (list[T.BaseTransform]): A list of PyTorch Geometric transforms
                to be applied to the dataset. These will be composed and applied
                sequentially to each data sample.
        """
        self.dataset = dataset
        # Compose all transforms into a single transform pipeline
        self.transform = T.Compose(transforms)

    def augment(self) -> Dataset:
        """
        Apply the composed transforms to all samples in the dataset.

        This method iterates through each data sample in the original dataset,
        applies the composed transformation pipeline, and creates a new dataset
        with the transformed samples.

        Returns:
            Dataset: A new dataset instance of the same class as the original,
            containing all the augmented data samples.
        """
        # Apply transforms to each data sample in the dataset and collect them in a list
        augmented_data_list = [self.transform(data) for data in tqdm(
            self.dataset, desc="Applying transforms", leave=False)]

        # Try to create a new dataset instance using the standard PyTorch Geometric interface
        try:
            # Standard PyTorch Geometric InMemoryDataset interface
            newDataset = self.dataset.__class__(
                root=self.dataset.root, transform=None, pre_transform=None)
            newDataset.data, newDataset.slices = self.dataset.collate(
                augmented_data_list)
        except (TypeError, AttributeError):
            # Handle custom datasets that don't follow the standard interface
            # Try to create using the original constructor signature
            try:
                # Check if the dataset has a simple constructor that takes a data_list
                newDataset = self.dataset.__class__(augmented_data_list)
            except TypeError:
                # Fallback: try to call constructor with no arguments and set data manually
                newDataset = self.dataset.__class__()
                newDataset.data_list = augmented_data_list

        return newDataset
