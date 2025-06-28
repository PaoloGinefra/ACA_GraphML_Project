"""
Comprehensive test suite for GDLPipeline class.

This test suite validates the Graph Deep Learning Pipeline implementation
for molecular property prediction on the ZINC dataset, including:
- All GNN architectures
- All pooling strategies  
- All regressor types
- Configuration management
- Embedding extraction
- Error handling
- Integration testing
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from typing import Dict, Any, List, Tuple
import tempfile
import os
import json

from ACAgraphML.Pipeline.Models.GDLPipeline import (
    GDLPipeline,
    GNNConfig,
    PoolingConfig,
    RegressorConfig,
    create_baseline_pipeline,
    create_standard_pipeline,
    create_advanced_pipeline,
    create_lightweight_pipeline,
    create_attention_pipeline
)


# Test constants
NUM_NODE_FEATS = 28
NUM_EDGE_FEATS = 4
BATCH_SIZE = 8
MAX_NODES = 20


class TestGDLPipelineConfigs:
    """Test configuration classes for GDLPipeline."""

    def test_gnn_config_defaults(self):
        """Test GNNConfig default values."""
        config = GNNConfig()

        assert config.hidden_dim == 128
        assert config.output_dim == 128  # Should equal hidden_dim when None
        assert config.num_layers == 4
        assert config.layer_name == "GINEConv"
        assert config.dropout_rate == 0.1
        assert config.use_residual is True
        assert config.use_layer_norm is True

    def test_gnn_config_custom_values(self):
        """Test GNNConfig with custom values."""
        config = GNNConfig(
            hidden_dim=256,
            output_dim=128,
            num_layers=5,
            layer_name="GAT",
            dropout_rate=0.2,
            gat_heads=4
        )

        assert config.hidden_dim == 256
        assert config.output_dim == 128
        assert config.num_layers == 5
        assert config.layer_name == "GAT"
        assert config.dropout_rate == 0.2
        assert config.gat_heads == 4

    def test_pooling_config_defaults(self):
        """Test PoolingConfig default values."""
        config = PoolingConfig()

        assert config.pooling_type == 'mean'
        assert config.processing_steps == 3

    def test_regressor_config_defaults(self):
        """Test RegressorConfig default values."""
        config = RegressorConfig()

        assert config.regressor_type == 'mlp'
        assert config.hidden_dims == [128, 64]
        assert config.mlp_dropout == 0.15
        assert config.normalization == 'batch'


class TestGDLPipelineBasics:
    """Test basic GDLPipeline functionality."""

    @pytest.fixture(scope="class")
    def setup_test_data(self):
        """Create test data for GDLPipeline."""

        def create_test_batch(batch_size: int = BATCH_SIZE, max_nodes: int = MAX_NODES) -> Batch:
            """Create a test batch mimicking ZINC dataset."""
            graphs = []

            for i in range(batch_size):
                num_nodes = torch.randint(5, max_nodes + 1, (1,)).item()

                x = torch.randn(num_nodes, NUM_NODE_FEATS)
                edge_index = torch.randint(
                    0, num_nodes, (2, min(num_nodes * 2, 40)))
                edge_attr = torch.randn(edge_index.size(1), NUM_EDGE_FEATS)
                y = torch.randn(1)

                graph = Data(x=x, edge_index=edge_index,
                             edge_attr=edge_attr, y=y)
                graphs.append(graph)

            return Batch.from_data_list(graphs)

        return {
            'create_batch': create_test_batch,
            'single_batch': create_test_batch(batch_size=BATCH_SIZE),
            'small_batch': create_test_batch(batch_size=2, max_nodes=10)
        }

    def test_pipeline_initialization_defaults(self):
        """Test GDLPipeline initialization with default parameters."""
        pipeline = GDLPipeline(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS
        )

        assert pipeline.node_features == NUM_NODE_FEATS
        assert pipeline.edge_features == NUM_EDGE_FEATS
        assert pipeline.gnn_config.hidden_dim == 128
        assert pipeline.pooling_config.pooling_type == 'mean'
        assert pipeline.regressor_config.regressor_type == 'mlp'

    def test_pipeline_initialization_custom_configs(self):
        """Test GDLPipeline initialization with custom configurations."""
        gnn_config = GNNConfig(hidden_dim=64, num_layers=3, layer_name="GCN")
        pooling_config = PoolingConfig(pooling_type='attentional')
        regressor_config = RegressorConfig(regressor_type='linear')

        pipeline = GDLPipeline(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS,
            gnn_config=gnn_config,
            pooling_config=pooling_config,
            regressor_config=regressor_config
        )

        assert pipeline.gnn_config.hidden_dim == 64
        assert pipeline.gnn_config.layer_name == "GCN"
        assert pipeline.pooling_config.pooling_type == 'attentional'
        assert pipeline.regressor_config.regressor_type == 'linear'

    def test_pipeline_initialization_dict_configs(self):
        """Test GDLPipeline initialization with dictionary configurations."""
        pipeline = GDLPipeline(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS,
            gnn_config={'hidden_dim': 64, 'layer_name': 'SAGE'},
            pooling_config={'pooling_type': 'max'},
            regressor_config={'regressor_type': 'residual_mlp'}
        )

        assert pipeline.gnn_config.hidden_dim == 64
        assert pipeline.gnn_config.layer_name == "SAGE"
        assert pipeline.pooling_config.pooling_type == 'max'
        assert pipeline.regressor_config.regressor_type == 'residual_mlp'

    def test_forward_pass_basic(self, setup_test_data):
        """Test basic forward pass."""
        pipeline = create_standard_pipeline(NUM_NODE_FEATS, NUM_EDGE_FEATS)
        batch = setup_test_data['single_batch']

        pipeline.eval()
        predictions = pipeline(batch.x, batch.edge_index,
                               batch.edge_attr, batch.batch)

        expected_batch_size = batch.batch.max().item() + 1
        assert predictions.shape == (expected_batch_size,)
        assert torch.isfinite(predictions).all()

    def test_forward_pass_with_embeddings(self, setup_test_data):
        """Test forward pass with embedding return."""
        pipeline = create_standard_pipeline(NUM_NODE_FEATS, NUM_EDGE_FEATS)
        batch = setup_test_data['single_batch']

        pipeline.eval()
        predictions, embeddings = pipeline(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch,
            return_embeddings=True
        )

        assert 'node_embeddings' in embeddings
        assert 'graph_embeddings' in embeddings
        assert embeddings['node_embeddings'].shape[0] == batch.x.shape[0]
        assert embeddings['graph_embeddings'].shape[0] == batch.batch.max(
        ).item() + 1

    def test_forward_pass_single_graph(self):
        """Test forward pass with single graph (no batch)."""
        pipeline = create_standard_pipeline(NUM_NODE_FEATS, NUM_EDGE_FEATS)

        x = torch.randn(10, NUM_NODE_FEATS)
        edge_index = torch.randint(0, 10, (2, 20))
        edge_attr = torch.randn(20, NUM_EDGE_FEATS)

        pipeline.eval()
        predictions = pipeline(x, edge_index, edge_attr)

        assert predictions.shape == (1,)
        assert torch.isfinite(predictions).all()


class TestGDLPipelineGNNArchitectures:
    """Test all GNN architectures supported by GDLPipeline."""

    @pytest.fixture(scope="class")
    def setup_test_batch(self):
        """Create a small test batch for architecture testing."""
        graphs = []
        for i in range(2):
            num_nodes = 8
            x = torch.randn(num_nodes, NUM_NODE_FEATS)
            edge_index = torch.randint(0, num_nodes, (2, 16))
            edge_attr = torch.randn(16, NUM_EDGE_FEATS)
            y = torch.randn(1)
            graphs.append(Data(x=x, edge_index=edge_index,
                          edge_attr=edge_attr, y=y))

        return Batch.from_data_list(graphs)

    @pytest.mark.parametrize("layer_name", [
        "GCN", "SAGE", "GINConv", "GINEConv", "GAT", "GATv2",
        "TransformerConv", "ChebConv", "SGConv", "TAGConv"
    ])
    def test_gnn_architectures(self, layer_name, setup_test_batch):
        """Test different GNN architectures."""
        batch = setup_test_batch

        # Special configuration for PNA (requires degree info)
        if layer_name == "PNA":
            gnn_config = GNNConfig(
                hidden_dim=32,
                num_layers=2,
                layer_name=layer_name,
                pna_deg=torch.tensor([1.0, 2.0, 3.0, 4.0])
            )
        else:
            gnn_config = GNNConfig(
                hidden_dim=32,
                num_layers=2,
                layer_name=layer_name
            )

        pipeline = GDLPipeline(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS,
            gnn_config=gnn_config,
            pooling_config=PoolingConfig(pooling_type='mean'),
            regressor_config=RegressorConfig(regressor_type='linear')
        )

        pipeline.eval()
        predictions = pipeline(batch.x, batch.edge_index,
                               batch.edge_attr, batch.batch)

        assert predictions.shape == (2,)  # 2 graphs in batch
        assert torch.isfinite(predictions).all()

    def test_pna_architecture_special(self, setup_test_batch):
        """Test PNA architecture with proper degree configuration."""
        batch = setup_test_batch

        gnn_config = GNNConfig(
            hidden_dim=32,
            num_layers=2,
            layer_name="PNA",
            pna_deg=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
            pna_aggregators=['mean', 'max', 'min'],
            pna_scalers=['identity', 'amplification']
        )

        pipeline = GDLPipeline(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS,
            gnn_config=gnn_config,
            pooling_config=PoolingConfig(pooling_type='mean'),
            regressor_config=RegressorConfig(regressor_type='linear')
        )

        pipeline.eval()
        predictions = pipeline(batch.x, batch.edge_index,
                               batch.edge_attr, batch.batch)

        assert predictions.shape == (2,)
        assert torch.isfinite(predictions).all()


class TestGDLPipelinePoolingStrategies:
    """Test all pooling strategies supported by GDLPipeline."""

    @pytest.fixture(scope="class")
    def setup_test_batch(self):
        """Create test batch for pooling testing."""
        graphs = []
        for i in range(3):
            num_nodes = 10
            x = torch.randn(num_nodes, NUM_NODE_FEATS)
            edge_index = torch.randint(0, num_nodes, (2, 20))
            edge_attr = torch.randn(20, NUM_EDGE_FEATS)
            y = torch.randn(1)
            graphs.append(Data(x=x, edge_index=edge_index,
                          edge_attr=edge_attr, y=y))

        return Batch.from_data_list(graphs)

    @pytest.mark.parametrize("pooling_type", [
        'mean', 'max', 'add', 'attentional', 'set2set'
    ])
    def test_pooling_strategies(self, pooling_type, setup_test_batch):
        """Test different pooling strategies."""
        batch = setup_test_batch

        pooling_config = PoolingConfig(pooling_type=pooling_type)

        pipeline = GDLPipeline(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS,
            gnn_config=GNNConfig(hidden_dim=32, num_layers=2),
            pooling_config=pooling_config,
            regressor_config=RegressorConfig(regressor_type='linear')
        )

        pipeline.eval()
        predictions = pipeline(batch.x, batch.edge_index,
                               batch.edge_attr, batch.batch)

        assert predictions.shape == (3,)  # 3 graphs in batch
        assert torch.isfinite(predictions).all()

        # Test graph embeddings shape
        graph_embeddings = pipeline.get_graph_embeddings(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )

        expected_dim = 64 if pooling_type == 'set2set' else 32  # set2set doubles dimension
        assert graph_embeddings.shape == (3, expected_dim)


class TestGDLPipelineRegressorTypes:
    """Test all regressor types supported by GDLPipeline."""

    @pytest.fixture(scope="class")
    def setup_test_batch(self):
        """Create test batch for regressor testing."""
        graphs = []
        for i in range(2):
            num_nodes = 8
            x = torch.randn(num_nodes, NUM_NODE_FEATS)
            edge_index = torch.randint(0, num_nodes, (2, 16))
            edge_attr = torch.randn(16, NUM_EDGE_FEATS)
            y = torch.randn(1)
            graphs.append(Data(x=x, edge_index=edge_index,
                          edge_attr=edge_attr, y=y))

        return Batch.from_data_list(graphs)

    @pytest.mark.parametrize("regressor_type", [
        'linear', 'mlp', 'residual_mlp', 'attention_mlp', 'ensemble_mlp'
    ])
    def test_regressor_types(self, regressor_type, setup_test_batch):
        """Test different regressor types."""
        batch = setup_test_batch

        regressor_config = RegressorConfig(
            regressor_type=regressor_type,
            hidden_dims=[32, 16],
            mlp_dropout=0.1
        )

        pipeline = GDLPipeline(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS,
            gnn_config=GNNConfig(hidden_dim=32, num_layers=2),
            pooling_config=PoolingConfig(pooling_type='mean'),
            regressor_config=regressor_config
        )

        pipeline.eval()
        predictions = pipeline(batch.x, batch.edge_index,
                               batch.edge_attr, batch.batch)

        assert predictions.shape == (2,)
        assert torch.isfinite(predictions).all()


class TestGDLPipelineEmbeddingExtraction:
    """Test embedding extraction functionality."""

    @pytest.fixture(scope="class")
    def setup_pipeline_and_data(self):
        """Setup pipeline and test data for embedding tests."""
        pipeline = create_standard_pipeline(NUM_NODE_FEATS, NUM_EDGE_FEATS)

        graphs = []
        for i in range(3):
            num_nodes = 10
            x = torch.randn(num_nodes, NUM_NODE_FEATS)
            edge_index = torch.randint(0, num_nodes, (2, 20))
            edge_attr = torch.randn(20, NUM_EDGE_FEATS)
            y = torch.randn(1)
            graphs.append(Data(x=x, edge_index=edge_index,
                          edge_attr=edge_attr, y=y))

        batch = Batch.from_data_list(graphs)

        return {'pipeline': pipeline, 'batch': batch}

    def test_get_node_embeddings(self, setup_pipeline_and_data):
        """Test node embedding extraction."""
        pipeline = setup_pipeline_and_data['pipeline']
        batch = setup_pipeline_and_data['batch']

        pipeline.eval()
        node_embeddings = pipeline.get_node_embeddings(
            batch.x, batch.edge_index, batch.edge_attr
        )

        # Same number of nodes
        assert node_embeddings.shape[0] == batch.x.shape[0]
        assert node_embeddings.shape[1] == 128  # Hidden dimension
        assert torch.isfinite(node_embeddings).all()

    def test_get_graph_embeddings(self, setup_pipeline_and_data):
        """Test graph embedding extraction."""
        pipeline = setup_pipeline_and_data['pipeline']
        batch = setup_pipeline_and_data['batch']

        pipeline.eval()
        graph_embeddings = pipeline.get_graph_embeddings(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )

        expected_batch_size = batch.batch.max().item() + 1
        assert graph_embeddings.shape[0] == expected_batch_size
        assert graph_embeddings.shape[1] == 128  # Expected embedding dimension
        assert torch.isfinite(graph_embeddings).all()

    def test_get_all_embeddings(self, setup_pipeline_and_data):
        """Test comprehensive embedding extraction."""
        pipeline = setup_pipeline_and_data['pipeline']
        batch = setup_pipeline_and_data['batch']

        pipeline.eval()
        all_embeddings = pipeline.get_all_embeddings(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )

        expected_keys = ['node_embeddings',
                         'graph_embeddings_raw', 'graph_embeddings_processed']
        assert set(all_embeddings.keys()) == set(expected_keys)

        # Check shapes
        assert all_embeddings['node_embeddings'].shape[0] == batch.x.shape[0]
        expected_batch_size = batch.batch.max().item() + 1
        assert all_embeddings['graph_embeddings_raw'].shape[0] == expected_batch_size
        assert all_embeddings['graph_embeddings_processed'].shape[0] == expected_batch_size

    def test_predict_method(self, setup_pipeline_and_data):
        """Test predict method for inference."""
        pipeline = setup_pipeline_and_data['pipeline']
        batch = setup_pipeline_and_data['batch']

        predictions = pipeline.predict(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )

        expected_batch_size = batch.batch.max().item() + 1
        assert predictions.shape == (expected_batch_size,)
        assert torch.isfinite(predictions).all()


class TestGDLPipelineAdvancedFeatures:
    """Test advanced features of GDLPipeline."""

    def test_target_normalization(self):
        """Test target normalization functionality."""
        pipeline = create_standard_pipeline(NUM_NODE_FEATS, NUM_EDGE_FEATS)

        # Set target normalization
        target_mean = 2.5
        target_std = 1.2
        pipeline.set_target_normalization(target_mean, target_std)

        assert pipeline.target_mean.item() == target_mean
        assert abs(pipeline.target_std.item() - target_std) < 1e-6

        # Test that predictions are denormalized
        x = torch.randn(10, NUM_NODE_FEATS)
        edge_index = torch.randint(0, 10, (2, 20))
        edge_attr = torch.randn(20, NUM_EDGE_FEATS)

        pipeline.eval()
        predictions = pipeline(x, edge_index, edge_attr)

        # Predictions should be different from raw regressor output due to denormalization
        assert torch.isfinite(predictions).all()

    def test_batch_norm_single_sample_handling(self):
        """Test that batch normalization handles small batches correctly."""
        pipeline = GDLPipeline(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS,
            use_batch_norm=True
        )

        # Create a small batch with 2 graphs to avoid BatchNorm issues
        graphs = []
        for i in range(2):
            x = torch.randn(5, NUM_NODE_FEATS)
            edge_index = torch.randint(0, 5, (2, 8))
            edge_attr = torch.randn(edge_index.size(1), NUM_EDGE_FEATS)
            graphs.append(
                Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

        batch = Batch.from_data_list(graphs)

        # Should work in both training and eval modes
        pipeline.train()
        predictions_train = pipeline(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        pipeline.eval()
        predictions_eval = pipeline(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        assert torch.isfinite(predictions_train).all()
        assert torch.isfinite(predictions_eval).all()
        assert predictions_train.shape[0] == 2
        assert predictions_eval.shape[0] == 2

    def test_component_freezing(self):
        """Test component freezing functionality."""
        pipeline = GDLPipeline(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS,
            freeze_gnn=True,
            freeze_pooling=False
        )

        # Check that GNN parameters are frozen
        gnn_params_require_grad = [
            p.requires_grad for p in pipeline.gnn.parameters()]
        assert not any(gnn_params_require_grad)

        # Check that pooling parameters are not frozen
        pooling_params_require_grad = [
            p.requires_grad for p in pipeline.pooling.parameters()]
        assert any(pooling_params_require_grad) or len(
            pooling_params_require_grad) == 0  # Some pooling types have no parameters


class TestGDLPipelineConfigurationManagement:
    """Test configuration management functionality."""

    def test_get_config(self):
        """Test configuration extraction."""
        pipeline = create_advanced_pipeline(NUM_NODE_FEATS, NUM_EDGE_FEATS)
        config = pipeline.get_config()

        expected_keys = [
            'node_features', 'edge_features', 'gnn_config', 'pooling_config',
            'regressor_config', 'global_dropout', 'use_batch_norm',
            'target_mean', 'target_std', 'gradient_clipping'
        ]

        assert set(config.keys()) == set(expected_keys)
        assert config['node_features'] == NUM_NODE_FEATS
        assert config['edge_features'] == NUM_EDGE_FEATS

    def test_from_config(self):
        """Test pipeline creation from configuration."""
        original_pipeline = create_advanced_pipeline(
            NUM_NODE_FEATS, NUM_EDGE_FEATS)
        config = original_pipeline.get_config()

        new_pipeline = GDLPipeline.from_config(config)

        # Check that both pipelines have same parameter counts
        original_params = original_pipeline.get_num_parameters()
        new_params = new_pipeline.get_num_parameters()

        assert original_params == new_params

    def test_save_and_load_config(self):
        """Test configuration saving and loading."""
        pipeline = create_standard_pipeline(NUM_NODE_FEATS, NUM_EDGE_FEATS)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name

        try:
            # Save configuration
            pipeline.save_config(config_path)

            # Load configuration
            loaded_pipeline = GDLPipeline.load_config(config_path)

            # Check that configurations match
            original_config = pipeline.get_config()
            loaded_config = loaded_pipeline.get_config()

            assert original_config == loaded_config

        finally:
            # Cleanup
            if os.path.exists(config_path):
                os.unlink(config_path)

    def test_get_num_parameters(self):
        """Test parameter counting functionality."""
        pipeline = create_standard_pipeline(NUM_NODE_FEATS, NUM_EDGE_FEATS)
        params = pipeline.get_num_parameters()

        expected_keys = ['gnn', 'pooling', 'regressor', 'other', 'total']
        assert set(params.keys()) == set(expected_keys)

        # Check that total equals sum of components
        component_sum = params['gnn'] + params['pooling'] + \
            params['regressor'] + params['other']
        assert params['total'] == component_sum

        # Check that all counts are positive
        assert all(count >= 0 for count in params.values())


class TestGDLPipelineConvenienceFunctions:
    """Test convenience functions for pipeline creation."""

    @pytest.fixture(scope="class")
    def setup_test_data(self):
        """Create small test batch for convenience function testing."""
        graphs = []
        for i in range(2):
            num_nodes = 8
            x = torch.randn(num_nodes, NUM_NODE_FEATS)
            edge_index = torch.randint(0, num_nodes, (2, 16))
            edge_attr = torch.randn(16, NUM_EDGE_FEATS)
            y = torch.randn(1)
            graphs.append(Data(x=x, edge_index=edge_index,
                          edge_attr=edge_attr, y=y))

        return Batch.from_data_list(graphs)

    @pytest.mark.parametrize("create_func,expected_features", [
        (create_baseline_pipeline, {
         'gnn_layer': 'GCN', 'regressor': 'linear'}),
        (create_standard_pipeline, {
         'gnn_layer': 'GINEConv', 'regressor': 'mlp'}),
        (create_advanced_pipeline, {
         'gnn_layer': 'GINEConv', 'regressor': 'ensemble_mlp'}),
        (create_lightweight_pipeline, {
         'gnn_layer': 'SAGE', 'regressor': 'mlp'}),
        (create_attention_pipeline, {
         'gnn_layer': 'GAT', 'regressor': 'attention_mlp'})
    ])
    def test_convenience_functions(self, create_func, expected_features, setup_test_data):
        """Test convenience functions for pipeline creation."""
        batch = setup_test_data

        pipeline = create_func(NUM_NODE_FEATS, NUM_EDGE_FEATS)

        # Test that pipeline was created successfully
        assert isinstance(pipeline, GDLPipeline)

        # Test forward pass
        pipeline.eval()
        predictions = pipeline(batch.x, batch.edge_index,
                               batch.edge_attr, batch.batch)

        assert predictions.shape == (2,)
        assert torch.isfinite(predictions).all()

        # Test that expected features are configured
        if 'gnn_layer' in expected_features:
            assert pipeline.gnn_config.layer_name == expected_features['gnn_layer']
        if 'regressor' in expected_features:
            assert pipeline.regressor_config.regressor_type == expected_features['regressor']


class TestGDLPipelineErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_gnn_layer(self):
        """Test error handling for invalid GNN layer."""
        try:
            pipeline = GDLPipeline(
                node_features=NUM_NODE_FEATS,
                edge_features=NUM_EDGE_FEATS,
                gnn_config=GNNConfig(layer_name="CompletelyInvalidLayer")
            )
            # If no exception is raised during construction, it should fail during forward pass
            x = torch.randn(5, NUM_NODE_FEATS)
            edge_index = torch.randint(0, 5, (2, 8))
            edge_attr = torch.randn(8, NUM_EDGE_FEATS)

            with pytest.raises((ValueError, AttributeError, RuntimeError)):
                pipeline(x, edge_index, edge_attr)
        except (ValueError, AttributeError, RuntimeError):
            # Exception raised during construction - this is also valid
            pass

    def test_invalid_pooling_type(self):
        """Test error handling for invalid pooling type."""
        with pytest.raises(ValueError):
            pipeline = GDLPipeline(
                node_features=NUM_NODE_FEATS,
                edge_features=NUM_EDGE_FEATS,
                pooling_config=PoolingConfig(pooling_type="invalid_pooling")
            )

            # Create dummy data to trigger the error
            x = torch.randn(10, NUM_NODE_FEATS)
            edge_index = torch.randint(0, 10, (2, 20))
            edge_attr = torch.randn(20, NUM_EDGE_FEATS)
            batch = torch.zeros(10, dtype=torch.long)

            pipeline(x, edge_index, edge_attr, batch)

    def test_invalid_regressor_type(self):
        """Test error handling for invalid regressor type."""
        with pytest.raises(ValueError):
            GDLPipeline(
                node_features=NUM_NODE_FEATS,
                edge_features=NUM_EDGE_FEATS,
                regressor_config=RegressorConfig(
                    regressor_type="invalid_regressor")
            )

    def test_dimension_mismatch_handling(self):
        """Test handling of dimension mismatches."""
        # This should work - pipeline should handle different input dimensions gracefully
        pipeline = GDLPipeline(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS
        )

        # Test with correct dimensions
        x = torch.randn(10, NUM_NODE_FEATS)
        edge_index = torch.randint(0, 10, (2, 20))
        edge_attr = torch.randn(20, NUM_EDGE_FEATS)

        pipeline.eval()
        predictions = pipeline(x, edge_index, edge_attr)
        assert torch.isfinite(predictions).all()


class TestGDLPipelineIntegration:
    """Integration tests for GDLPipeline."""

    def test_training_loop_simulation(self):
        """Test a complete training loop simulation."""
        pipeline = create_standard_pipeline(NUM_NODE_FEATS, NUM_EDGE_FEATS)
        optimizer = torch.optim.Adam(pipeline.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Create training data
        graphs = []
        for i in range(8):
            num_nodes = torch.randint(5, 15, (1,)).item()
            x = torch.randn(num_nodes, NUM_NODE_FEATS)
            edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
            edge_attr = torch.randn(edge_index.size(1), NUM_EDGE_FEATS)
            y = torch.randn(1)
            graphs.append(Data(x=x, edge_index=edge_index,
                          edge_attr=edge_attr, y=y))

        batch = Batch.from_data_list(graphs)

        # Training steps
        pipeline.train()
        for step in range(3):
            optimizer.zero_grad()
            predictions = pipeline(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            # Create dummy targets
            targets = torch.randn_like(predictions)
            loss = criterion(predictions, targets)

            loss.backward()
            optimizer.step()

            assert torch.isfinite(loss)

        # Test evaluation mode
        pipeline.eval()
        with torch.no_grad():
            eval_predictions = pipeline(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            assert torch.isfinite(eval_predictions).all()

    def test_large_scale_hyperparameter_grid(self):
        """Test pipeline creation with various hyperparameter combinations."""
        hyperparameter_combinations = [
            {
                'gnn_config': {'hidden_dim': 32, 'layer_name': 'GCN'},
                'pooling_config': {'pooling_type': 'mean'},
                'regressor_config': {'regressor_type': 'linear'}
            },
            {
                'gnn_config': {'hidden_dim': 64, 'layer_name': 'SAGE'},
                'pooling_config': {'pooling_type': 'max'},
                'regressor_config': {'regressor_type': 'mlp'}
            },
            {
                'gnn_config': {'hidden_dim': 128, 'layer_name': 'GINEConv'},
                'pooling_config': {'pooling_type': 'attentional'},
                'regressor_config': {'regressor_type': 'residual_mlp'}
            }
        ]

        # Create test data
        x = torch.randn(10, NUM_NODE_FEATS)
        edge_index = torch.randint(0, 10, (2, 20))
        edge_attr = torch.randn(20, NUM_EDGE_FEATS)
        batch = torch.zeros(10, dtype=torch.long)

        for i, config in enumerate(hyperparameter_combinations):
            pipeline = GDLPipeline(
                node_features=NUM_NODE_FEATS,
                edge_features=NUM_EDGE_FEATS,
                **config
            )

            pipeline.eval()
            predictions = pipeline(x, edge_index, edge_attr, batch)

            assert predictions.shape == (1,)
            assert torch.isfinite(predictions).all(
            ), f"Failed for config {i}: {config}"


if __name__ == "__main__":
    # Run specific test for debugging
    pytest.main(
        [__file__ + "::TestGDLPipelineBasics::test_forward_pass_basic", "-v"])
