from ACAgraphML.Transforms import AddMasterNode
import torch
from torch_geometric.data import Data
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_add_master_node_basic():
    """Test the AddMasterNode transform with basic functionality."""
    # Create a simple test graph
    # 3 nodes with 2D features
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float)

    # 2 edges: 0->1 and 1->2
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()

    # Edge features (2D)
    edge_attr = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float)

    # Create data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Test case 1: Zero initialization
    transform_zero = AddMasterNode(
        node_feature_init='zero', edge_feature_init='zero')
    transformed_data_zero = transform_zero(data.clone())

    # Assertions for zero initialization
    assert transformed_data_zero.x.size(
        0) == 4, "Should have 4 nodes (3 original + 1 master)"
    assert transformed_data_zero.edge_index.size(
        1) == 8, "Should have 8 edges (2 original + 6 master connections)"
    assert transformed_data_zero.x.shape == (
        4, 2), "Node features should be (4, 2)"
    assert transformed_data_zero.edge_attr.shape == (
        8, 2), "Edge features should be (8, 2)"

    # Check master node features are zeros
    expected_master_zero = torch.zeros(2, dtype=torch.float)
    assert torch.allclose(
        transformed_data_zero.x[-1], expected_master_zero), "Master node should have zero features"

    # Check that original nodes are unchanged
    assert torch.allclose(
        transformed_data_zero.x[:3], data.x), "Original nodes should be unchanged"

    # Check that original edges are preserved
    assert torch.equal(
        transformed_data_zero.edge_index[:, :2], data.edge_index), "Original edges should be preserved"

    # Check master node connections (bidirectional: master->others and others->master)
    master_edges = transformed_data_zero.edge_index[:, 2:]  # Last 6 edges
    expected_master_edges = torch.tensor(
        [[3, 3, 3, 0, 1, 2], [0, 1, 2, 3, 3, 3]], dtype=torch.long)
    assert torch.equal(
        master_edges, expected_master_edges), "Master node edges should be correct"

    # Check that new edge features are zeros
    # Last 6 edge features
    new_edge_features = transformed_data_zero.edge_attr[2:]
    expected_edge_zeros = torch.zeros(6, 2, dtype=torch.float)
    assert torch.allclose(
        new_edge_features, expected_edge_zeros), "New edge features should be zeros"


def test_add_master_node_mean_init():
    """Test the AddMasterNode transform with mean initialization."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
    edge_attr = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    transform_mean = AddMasterNode(
        node_feature_init='mean', edge_feature_init='mean')
    transformed_data_mean = transform_mean(data.clone())

    # Assertions for mean initialization
    assert transformed_data_mean.x.size(
        0) == 4, "Should have 4 nodes (3 original + 1 master)"
    assert transformed_data_mean.edge_index.size(
        1) == 8, "Should have 8 edges (2 original + 6 master connections)"

    # Check master node features are mean of original nodes
    expected_master_mean = data.x.mean(dim=0)  # [3.0, 4.0]
    assert torch.allclose(
        transformed_data_mean.x[-1], expected_master_mean), "Master node should have mean features"

    # Check that new edge features are means of original edges
    expected_edge_mean = data.edge_attr.mean(dim=0)  # [0.2, 0.3]
    # Last 6 edge features
    new_edge_features_mean = transformed_data_mean.edge_attr[2:]
    expected_edge_means = expected_edge_mean.repeat(6, 1)
    assert torch.allclose(new_edge_features_mean,
                          expected_edge_means), "New edge features should be means"


def test_add_master_node_ones_init():
    """Test the AddMasterNode transform with ones initialization."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
    edge_attr = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    transform_ones = AddMasterNode(
        node_feature_init='ones', edge_feature_init='ones')
    transformed_data_ones = transform_ones(data.clone())

    # Assertions for ones initialization
    assert transformed_data_ones.x.size(
        0) == 4, "Should have 4 nodes (3 original + 1 master)"
    assert transformed_data_ones.edge_index.size(
        1) == 8, "Should have 8 edges (2 original + 6 master connections)"

    # Check master node features are ones
    expected_master_ones = torch.ones(2, dtype=torch.float)
    assert torch.allclose(
        transformed_data_ones.x[-1], expected_master_ones), "Master node should have ones features"

    # Check that new edge features are ones
    # Last 6 edge features
    new_edge_features_ones = transformed_data_ones.edge_attr[2:]
    expected_edge_ones = torch.ones(6, 2, dtype=torch.float)
    assert torch.allclose(new_edge_features_ones,
                          expected_edge_ones), "New edge features should be ones"


def test_add_master_node_no_edge_attr():
    """Test the AddMasterNode transform with graphs that have no edge attributes."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
    data_no_edge_attr = Data(x=x, edge_index=edge_index)  # No edge_attr

    transform_no_edge = AddMasterNode(
        node_feature_init='mean', edge_feature_init='mean')
    transformed_no_edge = transform_no_edge(data_no_edge_attr.clone())

    # Assertions for graph without edge attributes
    assert transformed_no_edge.x.size(0) == 4, "Should have 4 nodes"
    assert transformed_no_edge.edge_index.size(1) == 8, "Should have 8 edges"
    assert not hasattr(
        transformed_no_edge, 'edge_attr') or transformed_no_edge.edge_attr is None, "Should not have edge attributes"

    # Check master node features
    expected_master_mean = data_no_edge_attr.x.mean(dim=0)
    assert torch.allclose(
        transformed_no_edge.x[-1], expected_master_mean), "Master node should have mean features"


def test_add_master_node_single_node():
    """Test the AddMasterNode transform with a single node graph."""
    single_x = torch.tensor([[1.0, 2.0]], dtype=torch.float)
    single_edge_index = torch.empty((2, 0), dtype=torch.long)  # No edges
    single_data = Data(x=single_x, edge_index=single_edge_index)

    transform_single = AddMasterNode(
        node_feature_init='zero', edge_feature_init='zero')
    transformed_single = transform_single(single_data.clone())

    # Assertions for single node graph
    assert transformed_single.x.size(
        0) == 2, "Should have 2 nodes (1 original + 1 master)"
    assert transformed_single.edge_index.size(
        1) == 2, "Should have 2 edges (bidirectional master connection)"

    # Check the edges connect node 0 and master node 1
    expected_single_edges = torch.tensor([[1, 0], [0, 1]], dtype=torch.long)
    assert torch.equal(transformed_single.edge_index,
                       expected_single_edges), "Edges should connect master to single node"


def test_add_master_node():
    """Run all tests for the AddMasterNode transform with demonstration output."""

    # Create a demo graph for showing the behavior
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
    edge_attr = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    print("Original graph:")
    print(f"Number of nodes: {data.x.size(0)}")
    print(f"Number of edges: {data.edge_index.size(1)}")
    print(f"Node features shape: {data.x.shape}")
    print(f"Edge features shape: {data.edge_attr.shape}")
    print(f"Edge index:\n{data.edge_index}")
    print()

    # Demonstrate different initialization strategies
    transforms = [
        ('zero', AddMasterNode(node_feature_init='zero', edge_feature_init='zero')),
        ('mean', AddMasterNode(node_feature_init='mean', edge_feature_init='mean')),
        ('ones', AddMasterNode(node_feature_init='ones', edge_feature_init='ones'))
    ]

    for name, transform in transforms:
        transformed_data = transform(data.clone())

        print(f"After applying {transform}:")
        print(f"Number of nodes: {transformed_data.x.size(0)}")
        print(f"Number of edges: {transformed_data.edge_index.size(1)}")
        print(f"Node features shape: {transformed_data.x.shape}")
        print(f"Edge features shape: {transformed_data.edge_attr.shape}")
        print(f"Master node features: {transformed_data.x[-1]}")
        print(f"New edges (last 6): {transformed_data.edge_index[:, -6:]}")
        print()

    # Run all test functions
    test_add_master_node_basic()
    test_add_master_node_mean_init()
    test_add_master_node_ones_init()
    test_add_master_node_no_edge_attr()
    test_add_master_node_single_node()

    print("All comprehensive tests passed!")


if __name__ == "__main__":
    test_add_master_node()
