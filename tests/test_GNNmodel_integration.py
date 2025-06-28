"""
Integration tests for GNNModel with ZINC dataset.
These tests focus on end-to-end functionality and performance characteristics.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from ACAgraphML.Dataset import ZINC_Dataset
from ACAgraphML.Transforms import OneHotEncodeFeat
from ACAgraphML.Pipeline.Models.GNNmodel import GNNModel
import time


class TestGNNModelIntegration:
    """Integration tests for GNNModel focusing on real-world usage scenarios."""

    @pytest.fixture(scope="class")
    def zinc_data(self):
        """Load ZINC dataset for integration testing."""
        oneHotTransform = OneHotEncodeFeat(28)

        def transform(data):
            data = oneHotTransform(data)
            data.x = data.x.float()
            data.edge_attr = torch.nn.functional.one_hot(
                data.edge_attr.long(), num_classes=4
            ).float()
            return data

        # Load datasets
        train_dataset = ZINC_Dataset.SMALL_TRAIN.load(transform=transform)
        val_dataset = ZINC_Dataset.SMALL_VAL.load(transform=transform)

        # Create small subsets for faster testing
        train_subset = train_dataset[:200]  # Small subset for training test
        val_subset = val_dataset[:50]       # Small subset for validation

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'num_node_features': 28,
            'num_edge_features': 4
        }

    @pytest.mark.parametrize("layer_name,expected_performance", [
        ("GCN", "baseline"),           # Simple baseline
        ("SAGE", "good"),              # Good performance/efficiency balance
        ("GINEConv", "excellent"),     # Should be best for molecular graphs
        ("GAT", "good"),               # Good with attention
        ("TransformerConv", "good"),   # Modern attention-based
    ])
    def test_training_integration(self, zinc_data, layer_name, expected_performance):
        """Test that models can be trained end-to-end on ZINC data."""

        # Set seeds for reproducibility in tests
        torch.manual_seed(42)
        np.random.seed(42)

        # Model configuration based on layer complexity (more conservative learning rates)
        config = {
            "GCN": {"hidden": 32, "layers": 2, "lr": 0.005},
            "SAGE": {"hidden": 64, "layers": 3, "lr": 0.003},
            "GINEConv": {"hidden": 64, "layers": 4, "lr": 0.001},
            "GAT": {"hidden": 32, "layers": 2, "lr": 0.003},
            "TransformerConv": {"hidden": 32, "layers": 2, "lr": 0.003},
        }

        model_config = config[layer_name]
        supports_edges = layer_name in ["GINEConv", "GAT", "TransformerConv"]

        # Create model
        model = GNNModel(
            c_in=zinc_data['num_node_features'],
            c_hidden=model_config["hidden"],
            c_out=64,  # Node-level features
            num_layers=model_config["layers"],
            layer_name=layer_name,
            edge_dim=zinc_data['num_edge_features'] if supports_edges else None,
            dp_rate=0.1,
            use_residual=True,
            use_layer_norm=True
        )

        # Add final regression head
        regression_head = nn.Linear(64, 1)

        # Optimizer with weight decay for stability
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(regression_head.parameters()),
            lr=model_config["lr"],
            weight_decay=1e-4
        )

        criterion = nn.MSELoss()

        # Training loop
        model.train()
        regression_head.train()

        epoch_losses = []

        # Train for a few epochs
        for epoch in range(5):  # Increased to 5 epochs for more stable training
            epoch_loss = 0
            num_batches = 0

            for batch in zinc_data['train_loader']:
                optimizer.zero_grad()

                # Forward pass through GNN
                if supports_edges:
                    node_embeddings = model(
                        batch.x, batch.edge_index, batch.edge_attr)
                else:
                    node_embeddings = model(batch.x, batch.edge_index)

                # Pool to graph level
                graph_embeddings = global_mean_pool(
                    node_embeddings, batch.batch)

                # Regression prediction
                predictions = regression_head(graph_embeddings).squeeze()

                # Compute loss
                loss = criterion(predictions, batch.y)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches
            epoch_losses.append(avg_epoch_loss)
            print(f"Epoch {epoch+1}, Average Loss: {avg_epoch_loss:.4f}")

        # Check training stability - either loss decreases or stays reasonably stable
        initial_loss = epoch_losses[0]
        final_loss = epoch_losses[-1]

        # More robust criteria: check that we either improve OR the model is learning (not diverging)
        loss_improved = final_loss < initial_loss
        # Allow up to 50% increase from initial
        loss_stable = final_loss < initial_loss * 1.5
        # At some point, loss decreased by 20%
        min_loss_achieved = min(epoch_losses) < initial_loss * 0.8

        training_successful = loss_improved or (
            loss_stable and min_loss_achieved)

        assert training_successful, (
            f"Training failed for {layer_name}. "
            f"Initial: {initial_loss:.4f}, Final: {final_loss:.4f}, "
            f"Min: {min(epoch_losses):.4f}, Losses: {[f'{l:.3f}' for l in epoch_losses]}"
        )
        assert not torch.isnan(torch.tensor(
            final_loss)), f"Final loss should not be NaN for {layer_name}"

        # Test evaluation mode
        model.eval()
        regression_head.eval()

        with torch.no_grad():
            val_loss = 0
            num_val_batches = 0

            for batch in zinc_data['val_loader']:
                if supports_edges:
                    node_embeddings = model(
                        batch.x, batch.edge_index, batch.edge_attr)
                else:
                    node_embeddings = model(batch.x, batch.edge_index)

                graph_embeddings = global_mean_pool(
                    node_embeddings, batch.batch)
                predictions = regression_head(graph_embeddings).squeeze()
                loss = criterion(predictions, batch.y)

                val_loss += loss.item()
                num_val_batches += 1

            avg_val_loss = val_loss / num_val_batches
            print(f"Validation Loss for {layer_name}: {avg_val_loss:.4f}")

        # Validation loss should be reasonable
        assert avg_val_loss < 10.0, f"Validation loss too high for {layer_name}"
        assert not torch.isnan(torch.tensor(
            avg_val_loss)), f"Validation loss should not be NaN for {layer_name}"

    def test_inference_speed(self, zinc_data):
        """Test inference speed for different layer types."""
        layer_names = ["GCN", "SAGE", "GINEConv", "GAT"]

        # Get a sample batch
        sample_batch = next(iter(zinc_data['val_loader']))

        inference_times = {}

        for layer_name in layer_names:
            supports_edges = layer_name in ["GINEConv", "GAT"]

            model = GNNModel(
                c_in=zinc_data['num_node_features'],
                c_hidden=64,
                c_out=32,
                num_layers=3,
                layer_name=layer_name,
                edge_dim=zinc_data['num_edge_features'] if supports_edges else None,
                dp_rate=0.0  # No dropout for speed test
            )

            model.eval()

            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    if supports_edges:
                        _ = model(sample_batch.x, sample_batch.edge_index,
                                  sample_batch.edge_attr)
                    else:
                        _ = model(sample_batch.x, sample_batch.edge_index)

            # Time inference
            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):
                    if supports_edges:
                        _ = model(sample_batch.x, sample_batch.edge_index,
                                  sample_batch.edge_attr)
                    else:
                        _ = model(sample_batch.x, sample_batch.edge_index)

            end_time = time.time()
            avg_time = (end_time - start_time) / 10
            inference_times[layer_name] = avg_time

            print(f"{layer_name} average inference time: {avg_time:.4f}s")

        # All inference times should be reasonable (< 1 second for small batch)
        for layer_name, time_taken in inference_times.items():
            assert time_taken < 1.0, f"{layer_name} inference too slow: {time_taken:.4f}s"

    def test_memory_usage(self, zinc_data):
        """Test memory usage for different configurations."""
        import gc
        import psutil
        import os

        def get_memory_usage():
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB

        configs = [
            {"layer_name": "GCN", "hidden": 32, "layers": 2},
            {"layer_name": "GINEConv", "hidden": 64, "layers": 3},
            # GAT can be memory intensive
            {"layer_name": "GAT", "hidden": 32, "layers": 2},
        ]

        sample_batch = next(iter(zinc_data['val_loader']))

        for config in configs:
            # Clear memory
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            initial_memory = get_memory_usage()

            supports_edges = config["layer_name"] in ["GINEConv", "GAT"]

            model = GNNModel(
                c_in=zinc_data['num_node_features'],
                c_hidden=config["hidden"],
                c_out=32,
                num_layers=config["layers"],
                layer_name=config["layer_name"],
                edge_dim=zinc_data['num_edge_features'] if supports_edges else None,
                dp_rate=0.1
            )

            # Forward pass
            model.eval()
            with torch.no_grad():
                if supports_edges:
                    output = model(
                        sample_batch.x, sample_batch.edge_index, sample_batch.edge_attr)
                else:
                    output = model(sample_batch.x, sample_batch.edge_index)

            final_memory = get_memory_usage()
            memory_increase = final_memory - initial_memory

            print(
                f"{config['layer_name']} memory increase: {memory_increase:.2f} MB")

            # Memory increase should be reasonable (< 500 MB for small models)
            assert memory_increase < 500, f"{config['layer_name']} uses too much memory: {memory_increase:.2f} MB"

            # Clean up
            del model, output
            gc.collect()

    def test_batch_size_scaling(self, zinc_data):
        """Test that models handle different batch sizes correctly."""
        layer_name = "GINEConv"  # Use a reliable layer

        model = GNNModel(
            c_in=zinc_data['num_node_features'],
            c_hidden=32,
            c_out=16,
            num_layers=2,
            layer_name=layer_name,
            edge_dim=zinc_data['num_edge_features'],
            dp_rate=0.1
        )

        # Test different batch sizes
        batch_sizes = [1, 8, 16, 32]

        for batch_size in batch_sizes:
            loader = DataLoader(
                next(iter(zinc_data['train_loader'])
                     ).to_data_list()[:batch_size],
                batch_size=batch_size,
                shuffle=False
            )

            batch = next(iter(loader))

            model.eval()
            with torch.no_grad():
                output = model(batch.x, batch.edge_index, batch.edge_attr)

            # Check output shape is correct
            assert output.shape[0] == batch.x.shape[
                0], f"Node dimension mismatch for batch_size={batch_size}"
            assert output.shape[1] == 16, f"Feature dimension should be 16"
            assert not torch.isnan(output).any(
            ), f"NaN outputs for batch_size={batch_size}"

            print(f"Batch size {batch_size}: {output.shape} ✓")

    def test_molecular_property_prediction_workflow(self, zinc_data):
        """Test complete workflow for molecular property prediction."""

        # Use GINEConv as it's proven good for molecular graphs
        node_model = GNNModel(
            c_in=zinc_data['num_node_features'],
            c_hidden=64,
            c_out=32,
            num_layers=4,
            layer_name="GINEConv",
            edge_dim=zinc_data['num_edge_features'],
            dp_rate=0.1,
            use_residual=True,
            use_layer_norm=True
        )

        # Graph-level prediction head
        graph_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )

        # Test forward pass
        sample_batch = next(iter(zinc_data['train_loader']))

        node_model.eval()
        graph_head.eval()

        with torch.no_grad():
            # Get node embeddings
            node_embeddings = node_model(
                sample_batch.x,
                sample_batch.edge_index,
                sample_batch.edge_attr
            )

            # Pool to graph level
            graph_embeddings = global_mean_pool(
                node_embeddings, sample_batch.batch)

            # Predict molecular property
            predictions = graph_head(graph_embeddings).squeeze()

        # Check outputs
        num_graphs = sample_batch.batch.max().item() + 1
        assert predictions.shape == (
            num_graphs,), f"Should predict one value per graph"
        assert torch.isfinite(predictions).all(
        ), "All predictions should be finite"

        # Compare with actual targets
        targets = sample_batch.y[:num_graphs]  # Get targets for this batch
        mae = torch.abs(predictions - targets).mean()

        print(f"Sample MAE: {mae:.4f}")
        print(
            f"Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
        print(f"Target range: [{targets.min():.3f}, {targets.max():.3f}]")

        # MAE should be reasonable for untrained model (not too large)
        assert mae < 10.0, f"MAE too large for untrained model: {mae:.4f}"

    def test_different_pooling_strategies(self, zinc_data):
        """Test different graph pooling strategies for molecular property prediction."""
        from torch_geometric.nn import global_max_pool, global_add_pool

        # Create node embedding model
        model = GNNModel(
            c_in=zinc_data['num_node_features'],
            c_hidden=32,
            c_out=16,
            num_layers=2,
            layer_name="SAGE",
            dp_rate=0.1
        )

        sample_batch = next(iter(zinc_data['val_loader']))

        model.eval()
        with torch.no_grad():
            node_embeddings = model(sample_batch.x, sample_batch.edge_index)

        # Test different pooling strategies
        pooling_functions = {
            'mean': global_mean_pool,
            'max': global_max_pool,
            'sum': global_add_pool,
        }

        num_graphs = sample_batch.batch.max().item() + 1

        for pool_name, pool_func in pooling_functions.items():
            graph_embeddings = pool_func(node_embeddings, sample_batch.batch)

            assert graph_embeddings.shape == (num_graphs, 16), \
                f"{pool_name} pooling shape incorrect: {graph_embeddings.shape}"
            assert torch.isfinite(graph_embeddings).all(), \
                f"{pool_name} pooling produced non-finite values"

            print(f"{pool_name} pooling: {graph_embeddings.shape} ✓")


if __name__ == "__main__":
    # Manual test execution
    print("Running GNN Model Integration Tests...")

    test_instance = TestGNNModelIntegration()
    zinc_data = test_instance.zinc_data()

    print("\n1. Testing training integration...")
    try:
        test_instance.test_training_integration(zinc_data, "GCN", "baseline")
        print("✓ Training integration test passed")
    except Exception as e:
        print(f"✗ Training integration test failed: {e}")

    print("\n2. Testing inference speed...")
    try:
        test_instance.test_inference_speed(zinc_data)
        print("✓ Inference speed test passed")
    except Exception as e:
        print(f"✗ Inference speed test failed: {e}")

    print("\n3. Testing molecular property prediction workflow...")
    try:
        test_instance.test_molecular_property_prediction_workflow(zinc_data)
        print("✓ Molecular property prediction test passed")
    except Exception as e:
        print(f"✗ Molecular property prediction test failed: {e}")

    print("\nAll integration tests completed! Run with pytest for full suite.")
