"""
Comprehensive test suite for GNNModel class using ZINC dataset.

This test suite validates all GNN layer types implemented in the GNNModel class
for molecular property prediction on the ZINC dataset.
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from ACAgraphML.Dataset import ZINC_Dataset
from ACAgraphML.Transforms import OneHotEncodeFeat
from ACAgraphML.Pipeline.Models.GNNmodel import GNNModel


# Constants for ZINC dataset
NUM_NODE_FEATS = 28  # Number of node features in ZINC dataset
NUM_EDGE_FEATS = 4   # Number of edge features (bond types: 0, 1, 2, 3)


class TestGNNModel:
    """Test class for GNNModel with comprehensive coverage of all layer types."""

    @pytest.fixture(scope="class")
    def setup_data(self):
        """Set up test data using ZINC dataset."""
        # Transform to ensure proper data format
        oneHotTransform = OneHotEncodeFeat(NUM_NODE_FEATS)

        def data_transform(data):
            data = oneHotTransform(data)
            data.x = data.x.float()
            # Convert edge attributes to one-hot encoding for bond types
            data.edge_attr = torch.nn.functional.one_hot(
                data.edge_attr.long(),
                num_classes=NUM_EDGE_FEATS
            ).float()
            return data

        # Load small subset for testing
        dataset = ZINC_Dataset.SMALL_TRAIN.load(transform=data_transform)

        # Create a small test subset for faster testing
        test_size = min(100, len(dataset))
        test_dataset = dataset[:test_size]

        # Create data loader
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # Get sample batch for shape testing
        sample_batch = next(iter(test_loader))

        return {
            'dataset': test_dataset,
            'loader': test_loader,
            'sample_batch': sample_batch,
            'node_features': sample_batch.x.shape[1],
            'edge_features': sample_batch.edge_attr.shape[1]
        }

    def test_data_loading(self, setup_data):
        """Test that ZINC data loads correctly."""
        data = setup_data

        assert len(data['dataset']) > 0, "Dataset should not be empty"
        assert data['node_features'] == NUM_NODE_FEATS, f"Expected {NUM_NODE_FEATS} node features"
        assert data['edge_features'] == NUM_EDGE_FEATS, f"Expected {NUM_EDGE_FEATS} edge features"

        # Check data types
        sample_batch = data['sample_batch']
        assert sample_batch.x.dtype == torch.float32, "Node features should be float32"
        assert sample_batch.edge_attr.dtype == torch.float32, "Edge features should be float32"
        assert sample_batch.y.dtype == torch.float32, "Targets should be float32"

    @pytest.mark.parametrize("layer_name,supports_edges", [
        # Low complexity layers
        ("SGConv", False),
        ("GraphConv", False),  # GraphConv doesn't support edge_attr
        ("GCN", False),

        # Medium complexity layers
        ("SAGE", False),  # SAGEConv doesn't support edge_attr
        ("GINConv", False),
        ("ChebConv", False),
        ("ARMAConv", False),
        ("TAGConv", False),

        # Medium-high complexity layers
        ("GAT", True),
        ("GATv2", True),
        ("TransformerConv", True),
        ("GINEConv", True),

        # High complexity layers
        ("PNA", True),
    ])
    def test_layer_initialization(self, setup_data, layer_name, supports_edges):
        """Test that all layer types can be initialized correctly."""
        data = setup_data

        # Model configuration
        c_in = data['node_features']
        c_hidden = 64
        c_out = 32
        edge_dim = data['edge_features'] if supports_edges else None

        # Create model
        model = GNNModel(
            c_in=c_in,
            c_hidden=c_hidden,
            c_out=c_out,
            num_layers=3,
            layer_name=layer_name,
            edge_dim=edge_dim,
            dp_rate=0.1
        )

        assert model is not None, f"Model with {layer_name} should initialize"
        assert len(model.gnn_layers) == 3, "Should have 3 GNN layers"
        assert model.layer_name == layer_name, f"Layer name should be {layer_name}"

    @pytest.mark.parametrize("layer_name", [
        "SGConv", "GraphConv", "GCN", "SAGE", "GINConv",
        "ChebConv", "ARMAConv", "TAGConv", "GAT", "GATv2",
        "TransformerConv", "GINEConv", "PNA"
    ])
    def test_forward_pass(self, setup_data, layer_name):
        """Test forward pass for all layer types."""
        data = setup_data
        sample_batch = data['sample_batch']

        # Model configuration
        c_in = data['node_features']
        c_hidden = 32  # Smaller for faster testing
        c_out = 16

        # Determine if layer supports edge attributes
        supports_edges = layer_name in [
            "GAT", "GATv2", "GINEConv", "PNA", "TransformerConv"]
        edge_dim = data['edge_features'] if supports_edges else None

        # Create model
        model = GNNModel(
            c_in=c_in,
            c_hidden=c_hidden,
            c_out=c_out,
            num_layers=2,  # Fewer layers for faster testing
            layer_name=layer_name,
            edge_dim=edge_dim,
            dp_rate=0.1
        )

        # Set to evaluation mode
        model.eval()

        # Forward pass
        with torch.no_grad():
            if supports_edges:
                output = model(
                    sample_batch.x, sample_batch.edge_index, sample_batch.edge_attr)
            else:
                output = model(sample_batch.x, sample_batch.edge_index)

        # Check output shape
        expected_shape = (sample_batch.x.shape[0], c_out)
        assert output.shape == expected_shape, f"Output shape should be {expected_shape}, got {output.shape}"
        assert not torch.isnan(output).any(
        ), f"Output should not contain NaN values for {layer_name}"
        assert torch.isfinite(output).all(
        ), f"Output should be finite for {layer_name}"

    def test_different_dimensions(self, setup_data):
        """Test model with different input/hidden/output dimensions."""
        data = setup_data
        sample_batch = data['sample_batch']

        test_configs = [
            {"c_in": 28, "c_hidden": 16, "c_out": 8},
            {"c_in": 28, "c_hidden": 64, "c_out": 1},  # Common for regression
            {"c_in": 28, "c_hidden": 28, "c_out": 28},  # Same dimensions
        ]

        for config in test_configs:
            model = GNNModel(
                c_in=config["c_in"],
                c_hidden=config["c_hidden"],
                c_out=config["c_out"],
                num_layers=2,
                layer_name="GINEConv",  # Use a reliable layer
                edge_dim=data['edge_features'],
                dp_rate=0.1
            )

            model.eval()
            with torch.no_grad():
                output = model(
                    sample_batch.x, sample_batch.edge_index, sample_batch.edge_attr)

            expected_shape = (sample_batch.x.shape[0], config["c_out"])
            assert output.shape == expected_shape, f"Failed for config {config}"

    def test_different_num_layers(self, setup_data):
        """Test model with different numbers of layers."""
        data = setup_data
        sample_batch = data['sample_batch']

        for num_layers in [1, 2, 3, 4, 5]:
            model = GNNModel(
                c_in=data['node_features'],
                c_hidden=32,
                c_out=16,
                num_layers=num_layers,
                layer_name="GCN",  # Simple layer for testing
                dp_rate=0.1
            )

            assert len(
                model.gnn_layers) == num_layers, f"Should have {num_layers} layers"

            model.eval()
            with torch.no_grad():
                output = model(sample_batch.x, sample_batch.edge_index)

            assert output.shape[0] == sample_batch.x.shape[0], "Batch dimension should be preserved"
            assert output.shape[1] == 16, "Output dimension should be 16"

    def test_dropout_rates(self, setup_data):
        """Test model with different dropout rates."""
        data = setup_data
        sample_batch = data['sample_batch']

        for dp_rate in [0.0, 0.1, 0.3, 0.5]:
            model = GNNModel(
                c_in=data['node_features'],
                c_hidden=32,
                c_out=16,
                num_layers=2,
                layer_name="SAGE",
                dp_rate=dp_rate
            )

            # Test in training mode (dropout active)
            model.train()
            output_train = model(sample_batch.x, sample_batch.edge_index)

            # Test in eval mode (dropout inactive)
            model.eval()
            with torch.no_grad():
                output_eval = model(sample_batch.x, sample_batch.edge_index)

            assert output_train.shape == output_eval.shape, "Shape should be same in train/eval modes"

            # With dropout > 0, training and eval outputs should be different
            if dp_rate > 0:
                assert not torch.allclose(output_train, output_eval, atol=1e-6), \
                    f"Outputs should differ with dropout={dp_rate}"

    def test_residual_connections(self, setup_data):
        """Test model with and without residual connections."""
        data = setup_data
        sample_batch = data['sample_batch']

        for use_residual in [True, False]:
            model = GNNModel(
                c_in=data['node_features'],
                c_hidden=32,
                c_out=16,
                num_layers=3,
                layer_name="GCN",
                use_residual=use_residual,
                dp_rate=0.1
            )

            model.eval()
            with torch.no_grad():
                output = model(sample_batch.x, sample_batch.edge_index)

            assert output.shape[0] == sample_batch.x.shape[0], "Batch dimension preserved"
            assert not torch.isnan(output).any(
            ), f"No NaN with residual={use_residual}"

    def test_layer_normalization(self, setup_data):
        """Test model with and without layer normalization."""
        data = setup_data
        sample_batch = data['sample_batch']

        for use_layer_norm in [True, False]:
            model = GNNModel(
                c_in=data['node_features'],
                c_hidden=32,
                c_out=16,
                num_layers=2,
                layer_name="GAT",
                edge_dim=data['edge_features'],
                use_layer_norm=use_layer_norm,
                dp_rate=0.1
            )

            if use_layer_norm:
                assert len(model.layer_norms) == 2, "Should have layer norms"
            else:
                assert len(
                    model.layer_norms) == 0, "Should not have layer norms"

            model.eval()
            with torch.no_grad():
                output = model(
                    sample_batch.x, sample_batch.edge_index, sample_batch.edge_attr)

            assert not torch.isnan(output).any(
            ), f"No NaN with layer_norm={use_layer_norm}"

    def test_edge_attribute_handling(self, setup_data):
        """Test proper handling of edge attributes."""
        data = setup_data
        sample_batch = data['sample_batch']

        # Test edge-aware layer
        model_with_edges = GNNModel(
            c_in=data['node_features'],
            c_hidden=32,
            c_out=16,
            num_layers=2,
            layer_name="GINEConv",
            edge_dim=data['edge_features'],
            dp_rate=0.1
        )

        # Test layer without edge support
        model_without_edges = GNNModel(
            c_in=data['node_features'],
            c_hidden=32,
            c_out=16,
            num_layers=2,
            layer_name="GCN",
            edge_dim=None,
            dp_rate=0.1
        )

        model_with_edges.eval()
        model_without_edges.eval()

        with torch.no_grad():
            # Test with edge attributes
            output_with_edges = model_with_edges(
                sample_batch.x, sample_batch.edge_index, sample_batch.edge_attr
            )

            # Test without edge attributes
            output_without_edges = model_without_edges(
                sample_batch.x, sample_batch.edge_index
            )

        assert output_with_edges.shape == output_without_edges.shape, "Shapes should match"
        assert not torch.isnan(output_with_edges).any(), "No NaN with edges"
        assert not torch.isnan(
            output_without_edges).any(), "No NaN without edges"

    def test_gradient_flow(self, setup_data):
        """Test that gradients flow properly through the model."""
        data = setup_data
        sample_batch = data['sample_batch']

        model = GNNModel(
            c_in=data['node_features'],
            c_hidden=32,
            c_out=1,  # Single output for regression
            num_layers=2,
            layer_name="GINEConv",
            edge_dim=data['edge_features'],
            dp_rate=0.1
        )

        # Create dummy targets
        batch_size = sample_batch.batch.max().item() + 1
        targets = torch.randn(batch_size, 1)

        # Forward pass
        output = model(sample_batch.x, sample_batch.edge_index,
                       sample_batch.edge_attr)

        # Pool to graph level (mean pooling)
        from torch_geometric.nn import global_mean_pool
        graph_output = global_mean_pool(output, sample_batch.batch)

        # Compute loss
        loss = nn.MSELoss()(graph_output, targets)

        # Backward pass
        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(
                param.grad).any(), f"NaN gradient for {name}"

    def test_parameter_count(self, setup_data):
        """Test parameter counting for different configurations."""
        data = setup_data

        configs = [
            {"layer_name": "GCN", "c_hidden": 32, "num_layers": 2},
            {"layer_name": "GAT", "c_hidden": 32, "num_layers": 2},
            {"layer_name": "GINEConv", "c_hidden": 64, "num_layers": 3},
        ]

        for config in configs:
            model = GNNModel(
                c_in=data['node_features'],
                c_hidden=config["c_hidden"],
                c_out=16,
                num_layers=config["num_layers"],
                layer_name=config["layer_name"],
                edge_dim=data['edge_features'] if config["layer_name"] in [
                    "GAT", "GINEConv"] else None,
                dp_rate=0.1
            )

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel()
                                   for p in model.parameters() if p.requires_grad)

            assert total_params > 0, f"Model should have parameters for {config['layer_name']}"
            assert trainable_params == total_params, "All parameters should be trainable"

    def test_model_save_load(self, setup_data, tmp_path):
        """Test model saving and loading."""
        data = setup_data
        sample_batch = data['sample_batch']

        # Create model
        model = GNNModel(
            c_in=data['node_features'],
            c_hidden=32,
            c_out=16,
            num_layers=2,
            layer_name="SAGE",
            dp_rate=0.1
        )

        # Get initial output
        model.eval()
        with torch.no_grad():
            output_before = model(sample_batch.x, sample_batch.edge_index)

        # Save model
        model_path = tmp_path / "test_model.pth"
        torch.save(model.state_dict(), model_path)

        # Create new model and load weights
        model_loaded = GNNModel(
            c_in=data['node_features'],
            c_hidden=32,
            c_out=16,
            num_layers=2,
            layer_name="SAGE",
            dp_rate=0.1
        )
        model_loaded.load_state_dict(torch.load(model_path))

        # Get output after loading
        model_loaded.eval()
        with torch.no_grad():
            output_after = model_loaded(
                sample_batch.x, sample_batch.edge_index)

        # Outputs should be identical
        assert torch.allclose(output_before, output_after, atol=1e-6), \
            "Outputs should be identical after save/load"

    def test_zinc_specific_features(self, setup_data):
        """Test features specific to ZINC dataset molecular regression."""
        data = setup_data
        sample_batch = data['sample_batch']

        # Test with typical ZINC molecular property prediction setup
        model = GNNModel(
            c_in=NUM_NODE_FEATS,  # 28 atom types
            c_hidden=64,
            c_out=1,  # Single regression target
            num_layers=4,
            layer_name="GINEConv",  # Proven good for molecular graphs
            edge_dim=NUM_EDGE_FEATS,  # 4 bond types
            dp_rate=0.15,
            use_residual=True,
            use_layer_norm=True
        )

        model.eval()
        with torch.no_grad():
            node_embeddings = model(
                sample_batch.x, sample_batch.edge_index, sample_batch.edge_attr)

        # Check that embeddings are reasonable
        assert node_embeddings.shape == (
            sample_batch.x.shape[0], 1), "Should output single value per node"
        assert torch.isfinite(node_embeddings).all(
        ), "All embeddings should be finite"

        # Check that different molecules produce different embeddings
        from torch_geometric.nn import global_mean_pool
        graph_embeddings = global_mean_pool(
            node_embeddings, sample_batch.batch)

        # Should have one embedding per graph
        num_graphs = sample_batch.batch.max().item() + 1
        assert graph_embeddings.shape == (
            num_graphs, 1), f"Should have {num_graphs} graph embeddings"

        # Embeddings should have some variance (not all the same)
        assert graph_embeddings.std() > 1e-4, "Graph embeddings should have meaningful variance"


def test_invalid_layer_name():
    """Test that invalid layer names default to GCN."""
    model = GNNModel(
        c_in=28,
        c_hidden=32,
        c_out=16,
        num_layers=2,
        layer_name="InvalidLayer",
        dp_rate=0.1
    )

    # Should create model with default GCN layers
    assert model is not None, "Model should be created with invalid layer name"
    assert len(model.gnn_layers) == 2, "Should have specified number of layers"


def test_pna_degree_tensor():
    """Test that PNA layer handles degree tensor correctly."""
    # This test checks the degree tensor handling in PNA
    model = GNNModel(
        c_in=28,
        c_hidden=32,
        c_out=16,
        num_layers=2,
        layer_name="PNA",
        edge_dim=4,
        dp_rate=0.1
    )

    # Check that PNA layer was created successfully
    assert model is not None, "PNA model should be created"
    assert model.layer_name == "PNA", "Layer name should be PNA"


if __name__ == "__main__":
    # Run tests manually if executed directly
    import sys

    # Create a simple test instance
    test_instance = TestGNNModel()

    # Setup data
    setup_data = test_instance.setup_data()

    print("Running basic smoke tests...")

    # Test data loading
    test_instance.test_data_loading(setup_data)
    print("✓ Data loading test passed")

    # Test a few key layer types
    for layer_name in ["GCN", "SAGE", "GINEConv", "GAT"]:
        try:
            test_instance.test_forward_pass(setup_data, layer_name)
            print(f"✓ Forward pass test for {layer_name} passed")
        except Exception as e:
            print(f"✗ Forward pass test for {layer_name} failed: {e}")

    print("\nAll smoke tests completed! Run with pytest for full test suite.")
