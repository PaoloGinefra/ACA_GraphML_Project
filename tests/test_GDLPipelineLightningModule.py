"""
Comprehensive test suite for GDLPipelineLightningModule.

This test suite validates the PyTorch Lightning wrapper for GDLPipeline including:
- Module initialization with different configurations
- Training, validation, and testing steps
- Optimizer and scheduler configuration
- Loss functions and metrics
- Embedding extraction
- Target normalization
- Error handling and edge cases
- Integration with PyTorch Lightning trainer
"""

import pytest
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from typing import Dict, Any, List, Tuple
import tempfile
import os
import warnings

from ACAgraphML.Pipeline.LightningModules.GDLPipelineLightningModule import (
    GDLPipelineLightningModule,
    create_lightning_baseline,
    create_lightning_standard,
    create_lightning_advanced,
    create_lightning_lightweight,
    create_lightning_attention,
    create_lightning_custom
)
from ACAgraphML.Pipeline.Models.GDLPipeline import (
    GNNConfig,
    PoolingConfig,
    RegressorConfig
)

# Test constants
NUM_NODE_FEATS = 28
NUM_EDGE_FEATS = 4
BATCH_SIZE = 8
MAX_NODES = 20


class TestGDLPipelineLightningModuleBasics:
    """Test basic GDLPipelineLightningModule functionality."""

    @pytest.fixture(scope="class")
    def setup_test_data(self):
        """Create test data for Lightning module testing."""

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

        def create_data_loaders():
            """Create train, val, test data loaders."""
            train_batch = create_test_batch(batch_size=16)
            val_batch = create_test_batch(batch_size=8)
            test_batch = create_test_batch(batch_size=8)

            # Convert to datasets and loaders
            train_loader = DataLoader([train_batch], batch_size=1)
            val_loader = DataLoader([val_batch], batch_size=1)
            test_loader = DataLoader([test_batch], batch_size=1)

            return train_loader, val_loader, test_loader

        return {
            'create_batch': create_test_batch,
            'create_loaders': create_data_loaders,
            'single_batch': create_test_batch(batch_size=BATCH_SIZE),
            'small_batch': create_test_batch(batch_size=2, max_nodes=10)
        }

    def test_lightning_module_initialization_defaults(self):
        """Test Lightning module initialization with default parameters."""
        model = GDLPipelineLightningModule(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS
        )

        assert model.node_features == NUM_NODE_FEATS
        assert model.edge_features == NUM_EDGE_FEATS
        assert model.monitor_metric == 'val_mae'
        assert model.optimizer_name == 'adamw'
        assert model.lr == 1e-3
        assert model.weight_decay == 1e-4
        assert model.lr_scheduler_name == 'cosine'
        assert model.gradient_clip_val == 1.0
        assert model.gradient_clip_algorithm == 'norm'

        # Check that pipeline was created
        assert hasattr(model, 'pipeline')
        assert model.pipeline is not None

    def test_lightning_module_initialization_custom(self):
        """Test Lightning module initialization with custom parameters."""
        model = GDLPipelineLightningModule(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS,
            pipeline_config="advanced",
            loss='mse',
            optimizer='adam',
            lr=5e-4,
            weight_decay=1e-5,
            lr_scheduler='plateau',
            gradient_clip_val=2.0,
            gradient_clip_algorithm='value',
            monitor_metric='val_rmse'
        )

        assert model.optimizer_name == 'adam'
        assert model.lr == 5e-4
        assert model.weight_decay == 1e-5
        assert model.lr_scheduler_name == 'plateau'
        assert model.gradient_clip_val == 2.0
        assert model.gradient_clip_algorithm == 'value'
        assert model.monitor_metric == 'val_rmse'

    def test_lightning_module_with_custom_pipeline_config(self):
        """Test Lightning module with custom pipeline configuration."""
        gnn_config = GNNConfig(
            hidden_dim=64,
            num_layers=3,
            layer_name="GAT",
            dropout_rate=0.2
        )

        pooling_config = PoolingConfig(pooling_type='attentional')

        regressor_config = RegressorConfig(
            regressor_type='residual_mlp',
            hidden_dims=[64, 32]
        )

        model = GDLPipelineLightningModule(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS,
            pipeline_config="custom",
            gnn_config=gnn_config,
            pooling_config=pooling_config,
            regressor_config=regressor_config
        )

        # Check that custom configuration was applied
        assert model.pipeline.gnn_config.hidden_dim == 64
        assert model.pipeline.gnn_config.layer_name == "GAT"
        assert model.pipeline.pooling_config.pooling_type == 'attentional'
        assert model.pipeline.regressor_config.regressor_type == 'residual_mlp'

    def test_forward_pass(self, setup_test_data):
        """Test forward pass through Lightning module."""
        model = GDLPipelineLightningModule(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS
        )

        batch = setup_test_data['single_batch']

        model.eval()
        with torch.no_grad():
            predictions = model(batch.x, batch.edge_index,
                                batch.edge_attr, batch.batch)

        expected_batch_size = batch.batch.max().item() + 1
        assert predictions.shape == (expected_batch_size,)
        assert torch.isfinite(predictions).all()

    def test_forward_pass_with_embeddings(self, setup_test_data):
        """Test forward pass with embedding extraction."""
        model = GDLPipelineLightningModule(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS
        )

        batch = setup_test_data['single_batch']

        model.eval()
        with torch.no_grad():
            predictions, embeddings = model(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                return_embeddings=True
            )

        assert 'node_embeddings' in embeddings
        assert 'graph_embeddings' in embeddings
        assert embeddings['node_embeddings'].shape[0] == batch.x.shape[0]
        assert embeddings['graph_embeddings'].shape[0] == batch.batch.max(
        ).item() + 1


class TestGDLPipelineLightningModuleLossFunctions:
    """Test different loss functions in Lightning module."""

    @pytest.mark.parametrize("loss_type", ['mae', 'mse', 'huber', 'smooth_l1'])
    def test_loss_functions(self, loss_type):
        """Test different loss function configurations."""
        model = GDLPipelineLightningModule(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS,
            loss=loss_type
        )

        # Test loss function was created correctly
        if loss_type == 'mae':
            assert isinstance(model.loss_fn, nn.L1Loss)
        elif loss_type == 'mse':
            assert isinstance(model.loss_fn, nn.MSELoss)
        elif loss_type == 'huber':
            assert isinstance(model.loss_fn, nn.HuberLoss)
        elif loss_type == 'smooth_l1':
            assert isinstance(model.loss_fn, nn.SmoothL1Loss)

    def test_invalid_loss_function(self):
        """Test error handling for invalid loss function."""
        with pytest.raises(ValueError, match="Unknown loss function"):
            GDLPipelineLightningModule(
                node_features=NUM_NODE_FEATS,
                edge_features=NUM_EDGE_FEATS,
                loss='invalid_loss'
            )


class TestGDLPipelineLightningModuleOptimizers:
    """Test optimizer configuration in Lightning module."""

    @pytest.mark.parametrize("optimizer_name", ['adam', 'adamw', 'sgd', 'rmsprop'])
    def test_optimizer_configuration(self, optimizer_name):
        """Test different optimizer configurations."""
        model = GDLPipelineLightningModule(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS,
            optimizer=optimizer_name,
            lr=1e-3,
            weight_decay=1e-4
        )

        optimizer = model.configure_optimizers()

        if isinstance(optimizer, dict):
            optimizer = optimizer['optimizer']

        if optimizer_name == 'adam':
            assert isinstance(optimizer, torch.optim.Adam)
        elif optimizer_name == 'adamw':
            assert isinstance(optimizer, torch.optim.AdamW)
        elif optimizer_name == 'sgd':
            assert isinstance(optimizer, torch.optim.SGD)
        elif optimizer_name == 'rmsprop':
            assert isinstance(optimizer, torch.optim.RMSprop)

        assert optimizer.param_groups[0]['lr'] == 1e-3
        assert optimizer.param_groups[0]['weight_decay'] == 1e-4

    def test_invalid_optimizer(self):
        """Test error handling for invalid optimizer."""
        model = GDLPipelineLightningModule(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS,
            optimizer='invalid_optimizer'
        )

        with pytest.raises(ValueError, match="Unknown optimizer"):
            model.configure_optimizers()


class TestGDLPipelineLightningModuleSchedulers:
    """Test learning rate scheduler configuration."""

    @pytest.mark.parametrize("scheduler_name", ['cosine', 'plateau', 'step', 'exponential', 'cosine_restarts'])
    def test_scheduler_configuration(self, scheduler_name):
        """Test different scheduler configurations."""
        model = GDLPipelineLightningModule(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS,
            lr_scheduler=scheduler_name,
            lr_scheduler_params={'T_max': 50, 'patience': 5, 'step_size': 10}
        )

        optimizer_config = model.configure_optimizers()

        if isinstance(optimizer_config, dict) and 'lr_scheduler' in optimizer_config:
            scheduler = optimizer_config['lr_scheduler']['scheduler']

            if scheduler_name == 'cosine':
                assert isinstance(
                    scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
            elif scheduler_name == 'plateau':
                assert isinstance(
                    scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
            elif scheduler_name == 'step':
                assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
            elif scheduler_name == 'exponential':
                assert isinstance(
                    scheduler, torch.optim.lr_scheduler.ExponentialLR)
            elif scheduler_name == 'cosine_restarts':
                assert isinstance(
                    scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)

    def test_no_scheduler(self):
        """Test configuration without scheduler."""
        model = GDLPipelineLightningModule(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS,
            lr_scheduler='none'
        )

        optimizer_config = model.configure_optimizers()

        # Should return just the optimizer
        assert isinstance(optimizer_config, torch.optim.Optimizer)

    def test_warmup_epochs(self):
        """Test learning rate warmup configuration."""
        model = GDLPipelineLightningModule(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS,
            warmup_epochs=5,
            lr_scheduler='cosine'
        )

        optimizer_config = model.configure_optimizers()

        assert isinstance(optimizer_config, dict)
        assert 'lr_scheduler' in optimizer_config

    def test_invalid_scheduler(self):
        """Test error handling for invalid scheduler."""
        model = GDLPipelineLightningModule(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS,
            lr_scheduler='invalid_scheduler'
        )

        with pytest.raises(ValueError, match="Unknown scheduler"):
            model.configure_optimizers()


class TestGDLPipelineLightningModuleTrainingSteps:
    """Test training, validation, and test steps."""

    @pytest.fixture
    def model_and_batch(self):
        """Create model and test batch."""
        model = GDLPipelineLightningModule(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS
        )

        # Create test batch
        graphs = []
        for i in range(4):
            num_nodes = 10
            x = torch.randn(num_nodes, NUM_NODE_FEATS)
            edge_index = torch.randint(0, num_nodes, (2, 20))
            edge_attr = torch.randn(20, NUM_EDGE_FEATS)
            y = torch.randn(1)
            graphs.append(Data(x=x, edge_index=edge_index,
                          edge_attr=edge_attr, y=y))

        batch = Batch.from_data_list(graphs)

        return model, batch

    def test_training_step(self, model_and_batch):
        """Test training step."""
        model, batch = model_and_batch

        # Test training step
        model.train()
        loss = model.training_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert torch.isfinite(loss)

    def test_validation_step(self, model_and_batch):
        """Test validation step."""
        model, batch = model_and_batch

        # Test validation step
        model.eval()
        outputs = model.validation_step(batch, 0)

        assert isinstance(outputs, dict)
        assert 'loss' in outputs
        assert 'mae' in outputs
        assert 'mse' in outputs
        assert 'rmse' in outputs
        assert 'r2_score' in outputs
        assert 'predictions' in outputs
        assert 'targets' in outputs

    def test_test_step(self, model_and_batch):
        """Test test step."""
        model, batch = model_and_batch

        # Test test step
        model.eval()
        outputs = model.test_step(batch, 0)

        assert isinstance(outputs, dict)
        assert 'loss' in outputs
        assert 'mae' in outputs
        assert 'mse' in outputs
        assert 'rmse' in outputs
        assert 'r2_score' in outputs
        assert 'predictions' in outputs
        assert 'targets' in outputs

    def test_shared_step_metrics(self, model_and_batch):
        """Test that shared step computes metrics correctly."""
        model, batch = model_and_batch

        outputs = model._shared_step(batch, 0, 'test')

        # Check that all metrics are computed
        assert torch.isfinite(outputs['loss'])
        assert torch.isfinite(outputs['mae'])
        assert torch.isfinite(outputs['mse'])
        assert torch.isfinite(outputs['rmse'])
        assert torch.isfinite(outputs['r2_score'])
        assert torch.isfinite(outputs['mean_abs_error'])
        assert torch.isfinite(outputs['std_abs_error'])
        assert torch.isfinite(outputs['max_abs_error'])

        # Check shapes
        assert outputs['predictions'].shape == outputs['targets'].shape


class TestGDLPipelineLightningModuleEmbeddings:
    """Test embedding extraction functionality."""

    @pytest.fixture
    def model_and_data(self):
        """Create model and test data."""
        model = GDLPipelineLightningModule(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS
        )

        # Single graph
        x = torch.randn(15, NUM_NODE_FEATS)
        edge_index = torch.randint(0, 15, (2, 30))
        edge_attr = torch.randn(30, NUM_EDGE_FEATS)
        batch = torch.zeros(15, dtype=torch.long)

        return model, x, edge_index, edge_attr, batch

    def test_get_embeddings_node(self, model_and_data):
        """Test node embedding extraction."""
        model, x, edge_index, edge_attr, batch = model_and_data

        node_embeddings = model.get_embeddings(
            x, edge_index, edge_attr,
            embedding_type='node'
        )

        assert node_embeddings.shape[0] == x.shape[0]
        assert torch.isfinite(node_embeddings).all()

    def test_get_embeddings_graph(self, model_and_data):
        """Test graph embedding extraction."""
        model, x, edge_index, edge_attr, batch = model_and_data

        graph_embeddings = model.get_embeddings(
            x, edge_index, edge_attr, batch,
            embedding_type='graph'
        )

        expected_batch_size = batch.max().item() + 1
        assert graph_embeddings.shape[0] == expected_batch_size
        assert torch.isfinite(graph_embeddings).all()

    def test_get_embeddings_all(self, model_and_data):
        """Test extraction of all embeddings."""
        model, x, edge_index, edge_attr, batch = model_and_data

        all_embeddings = model.get_embeddings(
            x, edge_index, edge_attr, batch,
            embedding_type='all'
        )

        assert isinstance(all_embeddings, dict)
        assert 'node_embeddings' in all_embeddings
        assert 'graph_embeddings_raw' in all_embeddings
        assert 'graph_embeddings_processed' in all_embeddings

    def test_predict_method(self, model_and_data):
        """Test predict method."""
        model, x, edge_index, edge_attr, batch = model_and_data

        predictions = model.predict(x, edge_index, edge_attr, batch)

        expected_batch_size = batch.max().item() + 1
        assert predictions.shape == (expected_batch_size,)
        assert torch.isfinite(predictions).all()

    def test_invalid_embedding_type(self, model_and_data):
        """Test error handling for invalid embedding type."""
        model, x, edge_index, edge_attr, batch = model_and_data

        with pytest.raises(ValueError, match="Unknown embedding_type"):
            model.get_embeddings(
                x, edge_index, edge_attr, batch,
                embedding_type='invalid_type'
            )


class TestGDLPipelineLightningModuleTargetNormalization:
    """Test target normalization functionality."""

    def test_target_normalization(self):
        """Test target normalization and denormalization."""
        target_mean = 2.5
        target_std = 1.2

        model = GDLPipelineLightningModule(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS,
            target_mean=target_mean,
            target_std=target_std
        )

        # Check that target normalization was set in pipeline
        assert model.pipeline.target_mean is not None
        assert model.pipeline.target_std is not None
        assert model.pipeline.target_mean.item() == target_mean
        assert abs(model.pipeline.target_std.item() - target_std) < 1e-6


class TestGDLPipelineLightningModuleConfiguration:
    """Test configuration management."""

    def test_get_pipeline_config(self):
        """Test pipeline configuration extraction."""
        model = GDLPipelineLightningModule(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS,
            pipeline_config="advanced"
        )

        config = model.get_pipeline_config()

        assert isinstance(config, dict)
        assert 'node_features' in config
        assert 'edge_features' in config
        assert 'gnn_config' in config
        assert 'pooling_config' in config
        assert 'regressor_config' in config

        assert config['node_features'] == NUM_NODE_FEATS
        assert config['edge_features'] == NUM_EDGE_FEATS

    def test_get_num_parameters(self):
        """Test parameter counting."""
        model = GDLPipelineLightningModule(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS
        )

        params = model.get_num_parameters()

        assert isinstance(params, dict)
        assert 'total' in params
        assert 'gnn' in params
        assert 'pooling' in params
        assert 'regressor' in params

        assert params['total'] > 0
        assert params['gnn'] > 0
        assert params['regressor'] > 0


class TestGDLPipelineLightningModuleConvenienceFunctions:
    """Test convenience functions for creating Lightning modules."""

    def test_create_lightning_baseline(self):
        """Test baseline Lightning module creation."""
        model = create_lightning_baseline(NUM_NODE_FEATS, NUM_EDGE_FEATS)

        assert isinstance(model, GDLPipelineLightningModule)
        assert model.node_features == NUM_NODE_FEATS
        assert model.edge_features == NUM_EDGE_FEATS

    def test_create_lightning_standard(self):
        """Test standard Lightning module creation."""
        model = create_lightning_standard(
            NUM_NODE_FEATS, NUM_EDGE_FEATS,
            lr=5e-4, weight_decay=1e-5
        )

        assert isinstance(model, GDLPipelineLightningModule)
        assert model.lr == 5e-4
        assert model.weight_decay == 1e-5

    def test_create_lightning_advanced(self):
        """Test advanced Lightning module creation."""
        model = create_lightning_advanced(NUM_NODE_FEATS, NUM_EDGE_FEATS)

        assert isinstance(model, GDLPipelineLightningModule)
        assert model.pipeline.gnn_config.layer_name == "GINEConv"

    def test_create_lightning_lightweight(self):
        """Test lightweight Lightning module creation."""
        model = create_lightning_lightweight(NUM_NODE_FEATS, NUM_EDGE_FEATS)

        assert isinstance(model, GDLPipelineLightningModule)
        assert model.pipeline.gnn_config.layer_name == "SAGE"

    def test_create_lightning_attention(self):
        """Test attention Lightning module creation."""
        model = create_lightning_attention(NUM_NODE_FEATS, NUM_EDGE_FEATS)

        assert isinstance(model, GDLPipelineLightningModule)
        assert model.pipeline.gnn_config.layer_name == "GAT"

    def test_create_lightning_custom(self):
        """Test custom Lightning module creation."""
        gnn_config = GNNConfig(hidden_dim=64, layer_name="SAGE")
        pooling_config = PoolingConfig(pooling_type='max')
        regressor_config = RegressorConfig(regressor_type='linear')

        model = create_lightning_custom(
            NUM_NODE_FEATS, NUM_EDGE_FEATS,
            gnn_config=gnn_config,
            pooling_config=pooling_config,
            regressor_config=regressor_config
        )

        assert isinstance(model, GDLPipelineLightningModule)
        assert model.pipeline.gnn_config.hidden_dim == 64
        assert model.pipeline.gnn_config.layer_name == "SAGE"
        assert model.pipeline.pooling_config.pooling_type == 'max'
        assert model.pipeline.regressor_config.regressor_type == 'linear'


class TestGDLPipelineLightningModuleErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_pipeline_config(self):
        """Test error handling for invalid pipeline config."""
        with pytest.raises(ValueError, match="Unknown pipeline_config"):
            GDLPipelineLightningModule(
                node_features=NUM_NODE_FEATS,
                edge_features=NUM_EDGE_FEATS,
                pipeline_config="invalid_config"
            )

    def test_gradient_clipping_configuration(self):
        """Test gradient clipping configuration."""
        model = GDLPipelineLightningModule(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS,
            gradient_clip_val=1.5,
            gradient_clip_algorithm='norm'
        )

        assert model.gradient_clip_val == 1.5
        assert model.gradient_clip_algorithm == 'norm'

    def test_label_smoothing_warnings(self):
        """Test that label smoothing warnings are issued for unsupported losses."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            model = GDLPipelineLightningModule(
                node_features=NUM_NODE_FEATS,
                edge_features=NUM_EDGE_FEATS,
                loss='mae',
                label_smoothing=0.1
            )

            # Check that warning was issued
            assert len(w) > 0
            assert "Label smoothing not supported" in str(w[0].message)


class TestGDLPipelineLightningModuleIntegration:
    """Integration tests with PyTorch Lightning trainer."""

    @pytest.fixture
    def setup_integration_test(self):
        """Setup for integration testing."""
        # Create model
        model = GDLPipelineLightningModule(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS,
            pipeline_config="lightweight"  # Use lightweight for faster testing
        )

        # Create minimal datasets
        def create_minimal_loader():
            graphs = []
            for i in range(4):
                num_nodes = 8
                x = torch.randn(num_nodes, NUM_NODE_FEATS)
                edge_index = torch.randint(0, num_nodes, (2, 16))
                edge_attr = torch.randn(16, NUM_EDGE_FEATS)
                y = torch.randn(1)
                graphs.append(Data(x=x, edge_index=edge_index,
                              edge_attr=edge_attr, y=y))

            batch = Batch.from_data_list(graphs)
            return DataLoader([batch], batch_size=1)

        train_loader = create_minimal_loader()
        val_loader = create_minimal_loader()
        test_loader = create_minimal_loader()

        return model, train_loader, val_loader, test_loader

    def test_trainer_fit(self, setup_integration_test):
        """Test training with PyTorch Lightning trainer."""
        model, train_loader, val_loader, test_loader = setup_integration_test

        # Create trainer
        trainer = Trainer(
            max_epochs=2,
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            accelerator='cpu',
            devices=1
        )

        # Test that training doesn't crash
        trainer.fit(model, train_loader, val_loader)

        # Verify model was trained
        assert trainer.current_epoch > 0

    def test_trainer_test(self, setup_integration_test):
        """Test testing with PyTorch Lightning trainer."""
        model, train_loader, val_loader, test_loader = setup_integration_test

        # Create trainer
        trainer = Trainer(
            max_epochs=1,
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            accelerator='cpu',
            devices=1
        )

        # Quick training
        trainer.fit(model, train_loader, val_loader)

        # Test
        results = trainer.test(model, test_loader)

        assert len(results) == 1
        assert 'test_loss' in results[0]
        assert 'test_mae' in results[0]

    def test_model_checkpointing(self, setup_integration_test):
        """Test model checkpointing functionality."""
        model, train_loader, val_loader, test_loader = setup_integration_test

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create trainer with checkpointing
            checkpoint_callback = ModelCheckpoint(
                dirpath=tmpdir,
                filename='test-{epoch}-{val_mae:.4f}',
                monitor='val_mae',
                mode='min'
            )

            trainer = Trainer(
                max_epochs=2,
                logger=False,
                enable_progress_bar=False,
                callbacks=[checkpoint_callback],
                accelerator='cpu',
                devices=1
            )

            # Train
            trainer.fit(model, train_loader, val_loader)

            # Check that checkpoint was created
            assert checkpoint_callback.best_model_path is not None
            assert os.path.exists(checkpoint_callback.best_model_path)

            # Test loading from checkpoint
            loaded_model = GDLPipelineLightningModule.load_from_checkpoint(
                checkpoint_callback.best_model_path
            )

            assert isinstance(loaded_model, GDLPipelineLightningModule)
            assert loaded_model.node_features == NUM_NODE_FEATS

    def test_early_stopping(self, setup_integration_test):
        """Test early stopping functionality."""
        model, train_loader, val_loader, test_loader = setup_integration_test

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=1,
            mode='min',
            verbose=False
        )

        trainer = Trainer(
            max_epochs=10,  # Set high so early stopping triggers
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            callbacks=[early_stop_callback],
            accelerator='cpu',
            devices=1
        )

        # Train (should stop early)
        trainer.fit(model, train_loader, val_loader)

        # Should have stopped before max_epochs
        assert trainer.current_epoch < 10


class TestGDLPipelineLightningModuleEpochEndHooks:
    """Test epoch end hooks and logging."""

    def test_validation_epoch_end(self):
        """Test validation epoch end processing."""
        model = GDLPipelineLightningModule(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS
        )

        # Simulate validation step outputs
        fake_outputs = []
        for i in range(3):
            fake_outputs.append({
                'predictions': torch.randn(5),
                'targets': torch.randn(5)
            })

        model.validation_step_outputs = fake_outputs

        # Test epoch end (should not crash)
        model.on_validation_epoch_end()

        # Check that outputs were cleared
        assert len(model.validation_step_outputs) == 0

    def test_test_epoch_end(self):
        """Test test epoch end processing."""
        model = GDLPipelineLightningModule(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS
        )

        # Simulate test step outputs
        fake_outputs = []
        for i in range(3):
            fake_outputs.append({
                'predictions': torch.randn(5),
                'targets': torch.randn(5)
            })

        model.test_step_outputs = fake_outputs

        # Test epoch end (should not crash)
        model.on_test_epoch_end()

        # Check that outputs were cleared
        assert len(model.test_step_outputs) == 0


if __name__ == "__main__":
    # Run specific test for debugging
    pytest.main(
        [__file__ + "::TestGDLPipelineLightningModuleBasics::test_lightning_module_initialization_defaults", "-v"])
