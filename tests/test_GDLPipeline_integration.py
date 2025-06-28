"""
Integration test suite for GDLPipeline with ZINC dataset.

This test suite validates the GDLPipeline integration with the actual ZINC dataset,
including:
- Data loading and preprocessing
- Training and evaluation workflows
- End-to-end pipeline validation
- Performance on real molecular data
- Model checkpointing and loading
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import tempfile
import os
from typing import Dict, Any, Optional

from ACAgraphML.Pipeline.Models.GDLPipeline import (
    GDLPipeline,
    GNNConfig,
    PoolingConfig,
    RegressorConfig,
    create_baseline_pipeline,
    create_standard_pipeline,
    create_advanced_pipeline
)
from ACAgraphML.Transforms import OneHotEncodeFeat

# Try to import ZINC dataset
try:
    from ACAgraphML.Dataset import ZINC_Dataset
    ZINC_AVAILABLE = True
except ImportError:
    ZINC_AVAILABLE = False


# Test constants
NUM_NODE_FEATS = 28
NUM_EDGE_FEATS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.mark.skipif(not ZINC_AVAILABLE, reason="ZINC dataset not available")
class TestGDLPipelineZINCIntegration:
    """Integration tests with real ZINC dataset."""

    @pytest.fixture(scope="class")
    def setup_zinc_data(self):
        """Set up ZINC dataset for testing."""
        # Transform to ensure proper data format
        oneHotTransform = OneHotEncodeFeat(NUM_NODE_FEATS)

        def data_transform(data):
            data = oneHotTransform(data)
            data.x = data.x.float()

            # Ensure edge attributes are properly formatted
            if data.edge_attr is not None:
                if data.edge_attr.dim() == 1:
                    # Convert to one-hot if needed
                    data.edge_attr = torch.nn.functional.one_hot(
                        data.edge_attr.long(),
                        num_classes=NUM_EDGE_FEATS
                    ).float()
                data.edge_attr = data.edge_attr.float()

            # Ensure target is float
            if data.y is not None:
                data.y = data.y.float()

            return data

        # Load ZINC subset for testing
        try:
            train_dataset = ZINC_Dataset(
                root='data/ZINC',
                subset=True,
                split='train',
                transform=data_transform
            )

            val_dataset = ZINC_Dataset(
                root='data/ZINC',
                subset=True,
                split='val',
                transform=data_transform
            )

            test_dataset = ZINC_Dataset(
                root='data/ZINC',
                subset=True,
                split='test',
                transform=data_transform
            )

            # Create small subsets for faster testing
            train_subset = train_dataset[:100]
            val_subset = val_dataset[:50]
            test_subset = test_dataset[:50]

            # Create data loaders
            train_loader = DataLoader(
                train_subset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
            test_loader = DataLoader(test_subset, batch_size=16, shuffle=False)

            return {
                'train_loader': train_loader,
                'val_loader': val_loader,
                'test_loader': test_loader,
                'num_train': len(train_subset),
                'num_val': len(val_subset),
                'num_test': len(test_subset)
            }

        except Exception as e:
            pytest.skip(f"Could not load ZINC dataset: {str(e)}")

    def test_zinc_data_compatibility(self, setup_zinc_data):
        """Test that GDLPipeline is compatible with ZINC data format."""
        data = setup_zinc_data
        pipeline = create_standard_pipeline(
            NUM_NODE_FEATS, NUM_EDGE_FEATS).to(DEVICE)

        # Test with a single batch
        batch = next(iter(data['train_loader']))
        batch = batch.to(DEVICE)

        pipeline.eval()
        with torch.no_grad():
            predictions = pipeline(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        # Verify output format
        expected_batch_size = batch.batch.max().item() + 1
        assert predictions.shape == (expected_batch_size,)
        assert torch.isfinite(predictions).all()

        # Verify target compatibility
        assert batch.y.shape[0] == expected_batch_size
        assert torch.isfinite(batch.y).all()

    def test_zinc_training_loop(self, setup_zinc_data):
        """Test complete training loop with ZINC data."""
        data = setup_zinc_data
        pipeline = create_baseline_pipeline(
            NUM_NODE_FEATS, NUM_EDGE_FEATS).to(DEVICE)

        # Setup training
        optimizer = optim.Adam(pipeline.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Training loop
        pipeline.train()
        train_losses = []

        for epoch in range(3):  # Just a few epochs for testing
            epoch_loss = 0.0
            num_batches = 0

            for batch in data['train_loader']:
                batch = batch.to(DEVICE)

                optimizer.zero_grad()
                predictions = pipeline(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = criterion(predictions, batch.y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)

            print(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.6f}")

        # Verify training progressed
        assert len(train_losses) == 3
        assert all(torch.isfinite(torch.tensor(loss)) for loss in train_losses)

        # Training should show some improvement or at least stability
        assert train_losses[-1] < train_losses[0] * \
            2, "Training loss increased too much"

    def test_zinc_evaluation_metrics(self, setup_zinc_data):
        """Test evaluation with standard metrics on ZINC data."""
        data = setup_zinc_data
        pipeline = create_standard_pipeline(
            NUM_NODE_FEATS, NUM_EDGE_FEATS).to(DEVICE)

        # Quick training to get a somewhat trained model
        optimizer = optim.Adam(pipeline.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        pipeline.train()
        for _ in range(2):
            for batch in data['train_loader']:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                predictions = pipeline(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = criterion(predictions, batch.y)
                loss.backward()
                optimizer.step()

        # Evaluation
        pipeline.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in data['val_loader']:
                batch = batch.to(DEVICE)
                predictions = pipeline(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch.y.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # Calculate metrics
        mse = np.mean((all_predictions - all_targets) ** 2)
        mae = np.mean(np.abs(all_predictions - all_targets))

        # Calculate correlation if there's variance in targets
        if np.std(all_targets) > 1e-6:
            correlation = np.corrcoef(all_predictions, all_targets)[0, 1]
        else:
            correlation = 0.0

        print(f"Evaluation Metrics:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Correlation: {correlation:.6f}")

        # Sanity checks
        assert np.isfinite(mse) and mse >= 0
        assert np.isfinite(mae) and mae >= 0
        assert np.isfinite(correlation)

    def test_zinc_embedding_analysis(self, setup_zinc_data):
        """Test embedding extraction and analysis on ZINC data."""
        data = setup_zinc_data
        pipeline = create_advanced_pipeline(
            NUM_NODE_FEATS, NUM_EDGE_FEATS).to(DEVICE)

        # Get a few batches for embedding analysis
        batches = [next(iter(data['val_loader'])) for _ in range(3)]

        pipeline.eval()

        all_node_embeddings = []
        all_graph_embeddings = []
        all_targets = []

        with torch.no_grad():
            for batch in batches:
                batch = batch.to(DEVICE)

                # Extract embeddings
                node_emb = pipeline.get_node_embeddings(
                    batch.x, batch.edge_index, batch.edge_attr
                )
                graph_emb = pipeline.get_graph_embeddings(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                )

                all_node_embeddings.append(node_emb.cpu())
                all_graph_embeddings.append(graph_emb.cpu())
                all_targets.append(batch.y.cpu())

        # Concatenate embeddings
        node_embeddings = torch.cat(all_node_embeddings, dim=0)
        graph_embeddings = torch.cat(all_graph_embeddings, dim=0)
        targets = torch.cat(all_targets, dim=0)

        # Verify embedding properties
        assert node_embeddings.shape[1] == 256  # Advanced pipeline hidden dim
        assert graph_embeddings.shape[1] == 256
        assert graph_embeddings.shape[0] == targets.shape[0]

        # Check embedding statistics
        node_std = node_embeddings.std(dim=0).mean()
        graph_std = graph_embeddings.std(dim=0).mean()

        print(f"Embedding Statistics:")
        print(f"  Node embeddings shape: {node_embeddings.shape}")
        print(f"  Graph embeddings shape: {graph_embeddings.shape}")
        print(f"  Node embedding std: {node_std:.4f}")
        print(f"  Graph embedding std: {graph_std:.4f}")

        # Embeddings should have reasonable variance
        assert node_std > 0.01, "Node embeddings have too little variance"
        assert graph_std > 0.01, "Graph embeddings have too little variance"
        assert torch.isfinite(node_embeddings).all()
        assert torch.isfinite(graph_embeddings).all()

    def test_zinc_model_checkpointing(self, setup_zinc_data):
        """Test model checkpointing and loading with ZINC data."""
        data = setup_zinc_data

        # Create and train a model
        pipeline = create_standard_pipeline(
            NUM_NODE_FEATS, NUM_EDGE_FEATS).to(DEVICE)
        optimizer = optim.Adam(pipeline.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Brief training
        pipeline.train()
        for _ in range(2):
            for batch in data['train_loader']:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                predictions = pipeline(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = criterion(predictions, batch.y)
                loss.backward()
                optimizer.step()

        # Save model and config
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model.pth')
            config_path = os.path.join(temp_dir, 'config.json')

            # Save
            torch.save(pipeline.state_dict(), model_path)
            pipeline.save_config(config_path)

            # Load
            new_pipeline = GDLPipeline.load_config(config_path).to(DEVICE)
            new_pipeline.load_state_dict(
                torch.load(model_path, map_location=DEVICE))

            # Test that loaded model produces same outputs
            test_batch = next(iter(data['val_loader'])).to(DEVICE)

            pipeline.eval()
            new_pipeline.eval()

            with torch.no_grad():
                original_pred = pipeline(test_batch.x, test_batch.edge_index,
                                         test_batch.edge_attr, test_batch.batch)
                loaded_pred = new_pipeline(test_batch.x, test_batch.edge_index,
                                           test_batch.edge_attr, test_batch.batch)

            # Should be very close (allowing for floating point precision)
            assert torch.allclose(original_pred, loaded_pred, atol=1e-6)

    @pytest.mark.parametrize("pipeline_type", [
        ("baseline", create_baseline_pipeline),
        ("standard", create_standard_pipeline),
        ("advanced", create_advanced_pipeline)
    ])
    def test_zinc_pipeline_variants(self, setup_zinc_data, pipeline_type):
        """Test different pipeline variants on ZINC data."""
        name, create_func = pipeline_type
        data = setup_zinc_data

        pipeline = create_func(NUM_NODE_FEATS, NUM_EDGE_FEATS).to(DEVICE)

        # Test forward pass
        batch = next(iter(data['train_loader'])).to(DEVICE)

        pipeline.eval()
        with torch.no_grad():
            predictions = pipeline(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        expected_batch_size = batch.batch.max().item() + 1
        assert predictions.shape == (expected_batch_size,)
        assert torch.isfinite(predictions).all()

        # Test parameter counts
        params = pipeline.get_num_parameters()
        print(f"{name.capitalize()} pipeline parameters: {params['total']:,}")

        assert params['total'] > 0
        assert params['gnn'] > 0
        assert params['regressor'] > 0

    def test_zinc_batch_size_robustness(self, setup_zinc_data):
        """Test pipeline robustness to different batch sizes."""
        data = setup_zinc_data
        pipeline = create_standard_pipeline(
            NUM_NODE_FEATS, NUM_EDGE_FEATS).to(DEVICE)

        # Test with different batch sizes
        batch_sizes = [1, 4, 8, 16]

        pipeline.eval()

        for batch_size in batch_sizes:
            # Create loader with specific batch size
            loader = DataLoader(
                [data['val_loader'].dataset[i] for i in range(
                    min(batch_size * 2, len(data['val_loader'].dataset)))],
                batch_size=batch_size,
                shuffle=False
            )

            batch = next(iter(loader)).to(DEVICE)

            with torch.no_grad():
                predictions = pipeline(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            expected_batch_size = batch.batch.max().item() + 1
            assert predictions.shape == (expected_batch_size,)
            assert torch.isfinite(predictions).all()

            print(
                f"Batch size {batch_size}: {expected_batch_size} graphs processed successfully")


@pytest.mark.skipif(ZINC_AVAILABLE, reason="ZINC dataset available, using real data tests")
class TestGDLPipelineMockZINC:
    """Mock ZINC tests when real dataset is not available."""

    def test_mock_zinc_compatibility(self):
        """Test pipeline with mock ZINC-like data."""
        from torch_geometric.data import Data, Batch

        # Create mock ZINC-like data
        graphs = []
        for i in range(10):
            num_nodes = torch.randint(10, 30, (1,)).item()
            x = torch.randn(num_nodes, NUM_NODE_FEATS)
            edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
            edge_attr = torch.randn(edge_index.size(1), NUM_EDGE_FEATS)
            y = torch.randn(1)  # Molecular property target

            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            graphs.append(graph)

        batch = Batch.from_data_list(graphs).to(DEVICE)

        # Test pipeline
        pipeline = create_standard_pipeline(
            NUM_NODE_FEATS, NUM_EDGE_FEATS).to(DEVICE)

        pipeline.eval()
        with torch.no_grad():
            predictions = pipeline(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        expected_batch_size = batch.batch.max().item() + 1
        assert predictions.shape == (expected_batch_size,)
        assert torch.isfinite(predictions).all()

        print("Mock ZINC test passed - pipeline compatible with ZINC-like data format")


class TestGDLPipelineRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_hyperparameter_optimization_scenario(self):
        """Test scenario for hyperparameter optimization."""
        from torch_geometric.data import Data, Batch

        # Simulate hyperparameter optimization loop
        hyperparams = [
            {'hidden_dim': 64, 'num_layers': 3, 'pooling': 'mean'},
            {'hidden_dim': 128, 'num_layers': 4, 'pooling': 'attentional'},
            {'hidden_dim': 256, 'num_layers': 5, 'pooling': 'set2set'}
        ]

        # Create mock data
        graphs = []
        for i in range(20):
            num_nodes = torch.randint(10, 25, (1,)).item()
            x = torch.randn(num_nodes, NUM_NODE_FEATS)
            edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
            edge_attr = torch.randn(edge_index.size(1), NUM_EDGE_FEATS)
            y = torch.randn(1)
            graphs.append(Data(x=x, edge_index=edge_index,
                          edge_attr=edge_attr, y=y))

        batch = Batch.from_data_list(graphs).to(DEVICE)

        results = []

        for i, hp in enumerate(hyperparams):
            # Create pipeline with hyperparameters
            pipeline = GDLPipeline(
                node_features=NUM_NODE_FEATS,
                edge_features=NUM_EDGE_FEATS,
                gnn_config=GNNConfig(
                    hidden_dim=hp['hidden_dim'],
                    num_layers=hp['num_layers'],
                    layer_name='GINEConv'
                ),
                pooling_config=PoolingConfig(pooling_type=hp['pooling']),
                regressor_config=RegressorConfig(regressor_type='mlp')
            ).to(DEVICE)

            # Quick training simulation
            optimizer = optim.Adam(pipeline.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            pipeline.train()
            for epoch in range(3):
                optimizer.zero_grad()
                predictions = pipeline(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = criterion(predictions, batch.y)
                loss.backward()
                optimizer.step()

            # Evaluation
            pipeline.eval()
            with torch.no_grad():
                final_predictions = pipeline(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                final_loss = criterion(final_predictions, batch.y).item()

            params = pipeline.get_num_parameters()

            result = {
                'config_id': i,
                'hyperparams': hp,
                'final_loss': final_loss,
                'parameters': params['total'],
                'success': True
            }
            results.append(result)

            print(
                f"Config {i+1}: Loss={final_loss:.6f}, Params={params['total']:,}")

        # Verify all configurations worked
        assert len(results) == len(hyperparams)
        assert all(r['success'] for r in results)
        assert all(np.isfinite(r['final_loss']) for r in results)

        # Find best configuration
        best_config = min(results, key=lambda x: x['final_loss'])
        print(
            f"Best configuration: {best_config['hyperparams']} (Loss: {best_config['final_loss']:.6f})")

    def test_production_deployment_scenario(self):
        """Test scenario for production deployment."""
        from torch_geometric.data import Data, Batch

        # Create a production-ready pipeline
        pipeline = create_standard_pipeline(NUM_NODE_FEATS, NUM_EDGE_FEATS)

        # Simulate model loading from checkpoint
        config = pipeline.get_config()
        loaded_pipeline = GDLPipeline.from_config(config)

        # Test single molecule prediction (production scenario)
        single_molecule = Data(
            x=torch.randn(15, NUM_NODE_FEATS),
            edge_index=torch.randint(0, 15, (2, 30)),
            edge_attr=torch.randn(30, NUM_EDGE_FEATS)
        )

        loaded_pipeline.eval()
        with torch.no_grad():
            prediction = loaded_pipeline.predict(
                single_molecule.x,
                single_molecule.edge_index,
                single_molecule.edge_attr
            )

        assert prediction.shape == (1,)
        assert torch.isfinite(prediction).all()

        # Test batch prediction
        molecules = [single_molecule for _ in range(5)]
        batch = Batch.from_data_list(molecules)

        with torch.no_grad():
            batch_predictions = loaded_pipeline.predict(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )

        assert batch_predictions.shape == (5,)
        assert torch.isfinite(batch_predictions).all()

        print("Production deployment scenario test passed")


if __name__ == "__main__":
    # Run integration tests
    if ZINC_AVAILABLE:
        pytest.main([__file__ + "::TestGDLPipelineZINCIntegration", "-v"])
    else:
        pytest.main([__file__ + "::TestGDLPipelineMockZINC", "-v"])
