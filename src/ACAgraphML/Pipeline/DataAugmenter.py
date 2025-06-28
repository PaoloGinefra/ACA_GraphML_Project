import torch_geometric.transforms as T
from torch_geometric.data import Dataset


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
        augmented_dataset = []

        # Apply transforms to each data sample in the dataset
        for data in self.dataset:
            augmented_data = self.transform(data)
            augmented_dataset.append(augmented_data)

        # Create a new dataset instance of the same class as the original
        # This preserves any dataset-specific metadata and functionality
        augmented_dataset = self.dataset.__class__(augmented_dataset)
        return augmented_dataset
