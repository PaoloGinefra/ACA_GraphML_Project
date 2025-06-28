from ACAgraphML.Transforms import AddMasterNode
import torch
from torch_geometric.data import Data
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_add_master_node():
    """Test the AddMasterNode transform."""

    # Create a simple test graph
    # 3 nodes with 2D features
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float)

    # 2 edges: 0->1 and 1->2
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()

    # Edge features (2D)
    edge_attr = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float)

    # Create data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    print("Original graph:")
    print(f"Number of nodes: {data.x.size(0)}")
    print(f"Number of edges: {data.edge_index.size(1)}")
    print(f"Node features shape: {data.x.shape}")
    print(f"Edge features shape: {data.edge_attr.shape}")
    print(f"Edge index:\n{data.edge_index}")
    print()

    # Apply the transform with different initialization strategies
    transforms = [
        AddMasterNode(node_feature_init='zero', edge_feature_init='zero'),
        AddMasterNode(node_feature_init='mean', edge_feature_init='mean'),
        AddMasterNode(node_feature_init='ones', edge_feature_init='ones')
    ]

    for i, transform in enumerate(transforms):
        transformed_data = transform(data.clone())

        print(f"After applying {transform}:")
        print(f"Number of nodes: {transformed_data.x.size(0)}")
        print(f"Number of edges: {transformed_data.edge_index.size(1)}")
        print(f"Node features shape: {transformed_data.x.shape}")
        print(f"Edge features shape: {transformed_data.edge_attr.shape}")
        print(f"Master node features: {transformed_data.x[-1]}")
        print(f"New edges (last 6): {transformed_data.edge_index[:, -6:]}")
        print()


if __name__ == "__main__":
    test_add_master_node()
