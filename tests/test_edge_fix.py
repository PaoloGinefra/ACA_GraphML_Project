"""
Quick test to verify GNN edge attribute fixes.
"""
import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, os.path.abspath(src_path))

try:
    from ACAgraphML.Dataset import ZINC_Dataset
    from ACAgraphML.Transforms import OneHotEncodeFeat
    from ACAgraphML.Pipeline.Models.GNNmodel import GNNModel
    import torch
    from torch_geometric.loader import DataLoader

    print("Testing GNN edge attribute fixes...")

    # Setup data
    oneHotTransform = OneHotEncodeFeat(28)

    def transform(data):
        data = oneHotTransform(data)
        data.x = data.x.float()
        data.edge_attr = torch.nn.functional.one_hot(
            data.edge_attr.long(), num_classes=4
        ).float()
        return data

    dataset = ZINC_Dataset.SMALL_TRAIN.load(transform=transform)
    small_dataset = dataset[:10]
    loader = DataLoader(small_dataset, batch_size=5, shuffle=False)
    sample_batch = next(iter(loader))

    # Test layers that should NOT support edge attributes
    print("\nTesting layers WITHOUT edge support:")
    no_edge_layers = ["GCN", "SAGE", "GraphConv", "SGConv", "GINConv"]

    for layer_name in no_edge_layers:
        try:
            model = GNNModel(
                c_in=28,
                c_hidden=16,
                c_out=8,
                num_layers=1,
                layer_name=layer_name,
                edge_dim=None,  # No edge support
                dp_rate=0.0
            )

            model.eval()
            with torch.no_grad():
                output = model(sample_batch.x, sample_batch.edge_index)

            print(f"✓ {layer_name}: Works without edge_attr")

        except Exception as e:
            print(f"✗ {layer_name}: Failed - {e}")

    # Test layers that SHOULD support edge attributes
    print("\nTesting layers WITH edge support:")
    edge_layers = ["GAT", "GATv2", "GINEConv", "PNA", "TransformerConv"]

    for layer_name in edge_layers:
        try:
            model = GNNModel(
                c_in=28,
                c_hidden=16,
                c_out=8,
                num_layers=1,
                layer_name=layer_name,
                edge_dim=4,  # With edge support
                dp_rate=0.0
            )

            model.eval()
            with torch.no_grad():
                output = model(
                    sample_batch.x, sample_batch.edge_index, sample_batch.edge_attr)

            print(f"✓ {layer_name}: Works with edge_attr")

        except Exception as e:
            print(f"✗ {layer_name}: Failed - {e}")

    print("\n✅ All edge attribute tests completed!")

except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
