from ACAgraphML.Transforms import AddMasterNode, SteadyStateTransform
from ACAgraphML.Pipeline.DataAugmenter import DataAugmenter
import pytest
import torch
import sys
import os
from torch_geometric.data import Data, Dataset
import torch_geometric.transforms as T

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


class SimpleTestDataset(Dataset):
    """A simple test dataset for testing DataAugmenter functionality."""

    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list
        self._indices = None

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def create_test_graph():
    """Create a simple test graph for testing."""
    # Create a simple test graph with 3 nodes and 2 edges
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
    edge_attr = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def test_data_augmenter_initialization():
    """Test that DataAugmenter initializes correctly."""
    # Create test dataset
    test_graph = create_test_graph()
    test_dataset = SimpleTestDataset([test_graph])

    # Create transforms
    transforms = [AddMasterNode(
        node_feature_init='zero', edge_feature_init='zero')]

    # Initialize DataAugmenter
    augmenter = DataAugmenter(test_dataset, transforms)

    # Check that the dataset and transform are stored correctly
    assert augmenter.dataset == test_dataset
    assert isinstance(augmenter.transform, T.Compose)


def test_data_augmenter_single_transform():
    """Test DataAugmenter with a single transform."""
    # Create test dataset with two graphs
    test_graph1 = create_test_graph()
    test_graph2 = create_test_graph()
    test_dataset = SimpleTestDataset([test_graph1, test_graph2])

    # Create DataAugmenter with AddMasterNode transform
    transforms = [AddMasterNode(
        node_feature_init='zero', edge_feature_init='zero')]
    augmenter = DataAugmenter(test_dataset, transforms)

    # Apply augmentation
    augmented_dataset = augmenter.augment()

    # Check that the augmented dataset has the same length
    assert len(augmented_dataset) == len(test_dataset)

    # Check that the augmented dataset is of the same class
    assert isinstance(augmented_dataset, SimpleTestDataset)

    # Check that each graph has been transformed (should have 4 nodes now)
    for i, data in enumerate(augmented_dataset):
        assert data.x.size(0) == 4  # 3 original + 1 master node
        # 2 original + 6 master connections
        assert data.edge_index.size(1) == 8

        # Check that master node features are zeros
        master_features = data.x[-1]  # Last node is master
        expected_zeros = torch.zeros(2, dtype=torch.float)
        assert torch.allclose(master_features, expected_zeros)


def test_data_augmenter_multiple_transforms():
    """Test DataAugmenter with multiple transforms."""
    # Create test dataset
    test_graph = create_test_graph()
    test_dataset = SimpleTestDataset([test_graph])

    # Create multiple transforms
    transforms = [
        AddMasterNode(node_feature_init='mean', edge_feature_init='mean'),
        # Small power for faster testing
        SteadyStateTransform(power=10, useEdgeWeights=False)
    ]

    # Create DataAugmenter
    augmenter = DataAugmenter(test_dataset, transforms)

    # Apply augmentation
    augmented_dataset = augmenter.augment()

    # Check basic properties
    assert len(augmented_dataset) == 1
    assert isinstance(augmented_dataset, SimpleTestDataset)

    # Get the transformed data
    transformed_data = augmented_dataset[0]

    # Check that AddMasterNode was applied (4 nodes)
    assert transformed_data.x.size(0) == 4

    # Check that SteadyStateTransform was applied (should have extra feature dimension)
    # Original: 2 features, SteadyState adds 1 more = 3 features total
    assert transformed_data.x.size(1) == 3


def test_data_augmenter_empty_transforms():
    """Test DataAugmenter with an empty transform list."""
    # Create test dataset
    test_graph = create_test_graph()
    test_dataset = SimpleTestDataset([test_graph])

    # Create DataAugmenter with no transforms
    augmenter = DataAugmenter(test_dataset, [])

    # Apply augmentation
    augmented_dataset = augmenter.augment()

    # Check that the dataset is unchanged
    assert len(augmented_dataset) == len(test_dataset)

    # Check that the data is identical (no transforms applied)
    original_data = test_dataset[0]
    augmented_data = augmented_dataset[0]

    assert torch.allclose(original_data.x, augmented_data.x)
    assert torch.equal(original_data.edge_index, augmented_data.edge_index)
    assert torch.allclose(original_data.edge_attr, augmented_data.edge_attr)


def test_data_augmenter_preserves_dataset_type():
    """Test that DataAugmenter preserves the original dataset class type."""

    class CustomDataset(SimpleTestDataset):
        """A custom dataset class to test type preservation."""

        def __init__(self, data_list):
            super().__init__(data_list)

        def custom_method(self):
            return "custom"

    # Create test dataset of custom type
    test_graph = create_test_graph()
    test_dataset = CustomDataset([test_graph])

    # Verify the custom method exists
    assert test_dataset.custom_method() == "custom"

    # Create DataAugmenter
    transforms = [AddMasterNode(node_feature_init='zero')]
    augmenter = DataAugmenter(test_dataset, transforms)

    # Apply augmentation
    augmented_dataset = augmenter.augment()

    # Check that the augmented dataset is of the same custom type
    assert isinstance(augmented_dataset, CustomDataset)
    assert augmented_dataset.custom_method() == "custom"


def test_data_augmenter_handles_different_graph_sizes():
    """Test DataAugmenter with graphs of different sizes."""
    # Create graphs with different numbers of nodes
    graph1 = Data(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_attr=torch.tensor([[0.1, 0.2]], dtype=torch.float)
    )

    graph2 = Data(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],
                       [7.0, 8.0]], dtype=torch.float),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        edge_attr=torch.tensor(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float)
    )

    test_dataset = SimpleTestDataset([graph1, graph2])

    # Apply AddMasterNode transform
    transforms = [AddMasterNode(
        node_feature_init='ones', edge_feature_init='ones')]
    augmenter = DataAugmenter(test_dataset, transforms)

    augmented_dataset = augmenter.augment()

    # Check that each graph was transformed correctly
    # Graph 1: 2 nodes -> 3 nodes (2 + 1 master)
    assert augmented_dataset[0].x.size(0) == 3

    # Graph 2: 4 nodes -> 5 nodes (4 + 1 master)
    assert augmented_dataset[1].x.size(0) == 5

    # Check that master node features are ones for both graphs
    for data in augmented_dataset:
        master_features = data.x[-1]  # Last node is master
        expected_ones = torch.ones(2, dtype=torch.float)
        assert torch.allclose(master_features, expected_ones)


def test_data_augmenter_error_handling():
    """Test that DataAugmenter handles edge cases and potential errors gracefully."""
    # Test with empty dataset
    empty_dataset = SimpleTestDataset([])
    transforms = [AddMasterNode()]
    augmenter = DataAugmenter(empty_dataset, transforms)

    augmented_dataset = augmenter.augment()
    assert len(augmented_dataset) == 0
    assert isinstance(augmented_dataset, SimpleTestDataset)


if __name__ == "__main__":
    # Run the tests
    test_data_augmenter_initialization()
    test_data_augmenter_single_transform()
    test_data_augmenter_multiple_transforms()
    test_data_augmenter_empty_transforms()
    test_data_augmenter_preserves_dataset_type()
    test_data_augmenter_handles_different_graph_sizes()
    test_data_augmenter_error_handling()

    print("All tests passed!")
