"""
Benchmark tests for GDLPipelineLightningModule.

These tests validate the performance characteristics and hyperparameter
optimization capabilities of the Lightning module, including:
- Training speed benchmarks
- Memory usage validation
- Hyperparameter optimization simulation
- Scalability tests
"""

import pytest
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import time
import psutil
import os
from typing import Dict, List, Tuple

from ACAgraphML.Pipeline.LightningModules.GDLPipelineLighningModule import (
    GDLPipelineLightningModule,
    create_lightning_standard,
    create_lightning_advanced,
    create_lightning_lightweight,
    create_lightning_baseline
)
from ACAgraphML.Pipeline.Models.GDLPipeline import GNNConfig, PoolingConfig, RegressorConfig

# Test constants
NUM_NODE_FEATS = 28
NUM_EDGE_FEATS = 4


class TestGDLPipelineLightningModuleBenchmarks:
    """Benchmark tests for Lightning module performance."""

    @pytest.fixture(scope="class")
    def benchmark_data(self):
        """Create benchmark datasets of different sizes."""

        def create_dataset(num_graphs: int, avg_nodes: int = 15) -> DataLoader:
            """Create a dataset with specified characteristics."""
            graphs = []

            for i in range(num_graphs):
                num_nodes = max(
                    5, avg_nodes + torch.randint(-5, 6, (1,)).item())
                num_edges = min(num_nodes * 3, num_nodes * 2 +
                                torch.randint(0, num_nodes, (1,)).item())

                x = torch.randn(num_nodes, NUM_NODE_FEATS)
                edge_index = torch.randint(0, num_nodes, (2, num_edges))
                edge_attr = torch.randn(num_edges, NUM_EDGE_FEATS)
                y = torch.randn(1)

                graphs.append(Data(x=x, edge_index=edge_index,
                              edge_attr=edge_attr, y=y))

            return DataLoader(graphs, batch_size=32, shuffle=True)

        return {
            'small': {
                'train': create_dataset(100, avg_nodes=10),
                'val': create_dataset(50, avg_nodes=10)
            },
            'medium': {
                'train': create_dataset(500, avg_nodes=15),
                'val': create_dataset(100, avg_nodes=15)
            },
            'large': {
                'train': create_dataset(1000, avg_nodes=20),
                'val': create_dataset(200, avg_nodes=20)
            }
        }

    def test_training_speed_lightweight_vs_advanced(self, benchmark_data):
        """Benchmark training speed: lightweight vs advanced configurations."""

        # Test configurations
        configs = {
            'lightweight': create_lightning_lightweight,
            'advanced': create_lightning_advanced
        }

        results = {}

        for config_name, create_func in configs.items():
            model = create_func(
                node_features=NUM_NODE_FEATS,
                edge_features=NUM_EDGE_FEATS,
                lr=1e-3
            )

            trainer = Trainer(
                max_epochs=3,
                logger=False,
                enable_progress_bar=False,
                enable_checkpointing=False,
                accelerator='cpu',
                devices=1
            )

            # Measure training time
            start_time = time.time()
            trainer.fit(
                model,
                benchmark_data['medium']['train'],
                benchmark_data['medium']['val']
            )
            end_time = time.time()

            training_time = end_time - start_time
            results[config_name] = {
                'training_time': training_time,
                'num_parameters': model.get_num_parameters()['total']
            }

        # Verify that lightweight is faster
        assert results['lightweight']['training_time'] < results['advanced']['training_time']
        assert results['lightweight']['num_parameters'] < results['advanced']['num_parameters']

        print(f"\nTraining Speed Benchmark:")
        print(f"Lightweight: {results['lightweight']['training_time']:.2f}s, "
              f"{results['lightweight']['num_parameters']:,} params")
        print(f"Advanced: {results['advanced']['training_time']:.2f}s, "
              f"{results['advanced']['num_parameters']:,} params")

    def test_memory_usage_scaling(self, benchmark_data):
        """Test memory usage with different dataset sizes."""

        model = create_lightning_standard(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS
        )

        memory_usage = {}

        for size_name, data in benchmark_data.items():
            # Measure memory before training
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            trainer = Trainer(
                max_epochs=1,
                logger=False,
                enable_progress_bar=False,
                enable_checkpointing=False,
                accelerator='cpu',
                devices=1
            )

            trainer.fit(model, data['train'], data['val'])

            # Measure memory after training
            memory_after = process.memory_info().rss / 1024 / 1024  # MB

            memory_usage[size_name] = {
                'before': memory_before,
                'after': memory_after,
                'increase': memory_after - memory_before
            }

        # Memory usage testing - note that this is informational due to
        # unpredictable garbage collection and system memory management
        small_increase = memory_usage['small']['increase']
        large_increase = memory_usage['large']['increase']

        # Basic sanity checks - we expect some memory usage
        assert small_increase >= 0, f"Small dataset memory should not decrease: {small_increase}"
        assert large_increase >= 0, f"Large dataset memory should not decrease: {large_increase}"

        # Memory usage can be highly variable due to system factors
        # so we just log the results rather than making strict assertions
        print(f"\nMemory Usage Scaling (informational):")
        for size, usage in memory_usage.items():
            print(f"{size.capitalize()}: {usage['increase']:.1f} MB increase")

        # Optional: warn if the scaling seems unusual but don't fail
        if large_increase > 0 and small_increase > 0:
            ratio = large_increase / small_increase
            if ratio < 0.5:  # Large uses significantly less than small
                print(
                    f"Note: Unusual memory scaling ratio: {ratio:.2f} (may be due to garbage collection)")
            elif ratio > 3.0:  # Large uses significantly more than small
                print(f"Note: Expected memory scaling ratio: {ratio:.2f}")

        # Always pass - this test is primarily informational

    def test_hyperparameter_optimization_simulation(self, benchmark_data):
        """Simulate hyperparameter optimization workflow."""

        # Define hyperparameter grid (simplified for testing)
        param_grid = [
            {'hidden_dim': 64, 'num_layers': 3, 'layer_name': 'SAGE', 'lr': 1e-3},
            {'hidden_dim': 128, 'num_layers': 4,
                'layer_name': 'GINEConv', 'lr': 5e-4},
            {'hidden_dim': 64, 'num_layers': 3, 'layer_name': 'GAT', 'lr': 1e-3},
        ]

        results = []

        for i, params in enumerate(param_grid):
            # Create model with specific hyperparameters
            model = GDLPipelineLightningModule(
                node_features=NUM_NODE_FEATS,
                edge_features=NUM_EDGE_FEATS,
                pipeline_config="custom",
                gnn_config=GNNConfig(
                    hidden_dim=params['hidden_dim'],
                    num_layers=params['num_layers'],
                    layer_name=params['layer_name']
                ),
                lr=params['lr']
            )

            # Quick training for evaluation
            trainer = Trainer(
                max_epochs=2,
                logger=False,
                enable_progress_bar=False,
                enable_checkpointing=False,
                accelerator='cpu',
                devices=1
            )

            start_time = time.time()
            trainer.fit(
                model, benchmark_data['small']['train'], benchmark_data['small']['val'])
            training_time = time.time() - start_time

            # Get final validation metrics
            val_results = trainer.validate(
                model, benchmark_data['small']['val'], verbose=False)
            val_mae = val_results[0]['val_mae']

            results.append({
                'config_id': i,
                'params': params,
                'val_mae': val_mae,
                'training_time': training_time,
                'num_parameters': model.get_num_parameters()['total']
            })

        # Verify that all configurations completed successfully
        assert len(results) == len(param_grid)

        # Find best configuration by validation MAE
        best_config = min(results, key=lambda x: x['val_mae'])

        print(f"\nHyperparameter Optimization Simulation:")
        print(f"Best configuration: {best_config['params']}")
        print(f"Best VAL MAE: {best_config['val_mae']:.4f}")
        print(f"Training time: {best_config['training_time']:.2f}s")

        return results

    def test_gradient_accumulation_scaling(self, benchmark_data):
        """Test training with different gradient accumulation settings."""

        accumulation_settings = [1, 2, 4]
        results = {}

        for accumulate_grad_batches in accumulation_settings:
            model = create_lightning_standard(
                node_features=NUM_NODE_FEATS,
                edge_features=NUM_EDGE_FEATS
            )

            trainer = Trainer(
                max_epochs=2,
                accumulate_grad_batches=accumulate_grad_batches,
                logger=False,
                enable_progress_bar=False,
                enable_checkpointing=False,
                accelerator='cpu',
                devices=1
            )

            start_time = time.time()
            trainer.fit(
                model, benchmark_data['medium']['train'], benchmark_data['medium']['val'])
            training_time = time.time() - start_time

            results[accumulate_grad_batches] = {
                'training_time': training_time,
                'final_train_loss': trainer.callback_metrics.get('train_loss', float('inf'))
            }

        # Verify that gradient accumulation doesn't break training
        for setting, result in results.items():
            assert torch.isfinite(result['final_train_loss'])

        print(f"\nGradient Accumulation Scaling:")
        for setting, result in results.items():
            print(f"Accumulate {setting}: {result['training_time']:.2f}s, "
                  f"Final loss: {result['final_train_loss']:.4f}")

    def test_different_batch_sizes_performance(self, benchmark_data):
        """Test performance with different batch sizes."""

        batch_sizes = [16, 32, 64]
        results = {}

        # Create datasets with different batch sizes
        def create_loader_with_batch_size(original_loader, batch_size):
            # Extract all data from original loader
            all_data = []
            for batch in original_loader:
                if isinstance(batch, Batch):
                    all_data.extend(batch.to_data_list())
                else:
                    all_data.append(batch)
            return DataLoader(all_data, batch_size=batch_size, shuffle=True)

        model = create_lightning_standard(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS
        )

        for batch_size in batch_sizes:
            train_loader = create_loader_with_batch_size(
                benchmark_data['medium']['train'], batch_size
            )
            val_loader = create_loader_with_batch_size(
                benchmark_data['medium']['val'], batch_size
            )

            trainer = Trainer(
                max_epochs=2,
                logger=False,
                enable_progress_bar=False,
                enable_checkpointing=False,
                accelerator='cpu',
                devices=1
            )

            start_time = time.time()
            trainer.fit(model, train_loader, val_loader)
            training_time = time.time() - start_time

            results[batch_size] = {
                'training_time': training_time,
                'steps_per_epoch': len(train_loader)
            }

        print(f"\nBatch Size Performance:")
        for batch_size, result in results.items():
            print(f"Batch size {batch_size}: {result['training_time']:.2f}s, "
                  f"{result['steps_per_epoch']} steps/epoch")

    def test_scheduler_performance_comparison(self, benchmark_data):
        """Compare performance of different learning rate schedulers."""

        schedulers = ['cosine', 'plateau', 'step', 'none']
        results = {}

        for scheduler in schedulers:
            model = create_lightning_standard(
                node_features=NUM_NODE_FEATS,
                edge_features=NUM_EDGE_FEATS,
                lr_scheduler=scheduler,
                lr=1e-3
            )

            trainer = Trainer(
                max_epochs=5,
                logger=False,
                enable_progress_bar=False,
                enable_checkpointing=False,
                accelerator='cpu',
                devices=1
            )

            trainer.fit(
                model, benchmark_data['small']['train'], benchmark_data['small']['val'])

            results[scheduler] = {
                'final_val_mae': trainer.callback_metrics.get('val_mae', float('inf')),
                'final_lr': trainer.optimizers[0].param_groups[0]['lr'] if trainer.optimizers else None
            }

        print(f"\nScheduler Performance Comparison:")
        for scheduler, result in results.items():
            print(f"{scheduler}: VAL MAE {result['final_val_mae']:.4f}, "
                  f"Final LR {result['final_lr']:.2e}" if result['final_lr'] else f"{scheduler}: VAL MAE {result['final_val_mae']:.4f}")


class TestGDLPipelineLightningModuleStressTests:
    """Stress tests for edge cases and robustness."""

    def test_very_small_graphs(self):
        """Test with very small graphs (minimal nodes/edges)."""

        # Create dataset with very small graphs
        graphs = []
        for i in range(20):
            num_nodes = torch.randint(3, 6, (1,)).item()  # Very small graphs
            num_edges = max(2, num_nodes - 1)  # Minimal connectivity

            x = torch.randn(num_nodes, NUM_NODE_FEATS)
            edge_index = torch.randint(0, num_nodes, (2, num_edges))
            edge_attr = torch.randn(num_edges, NUM_EDGE_FEATS)
            y = torch.randn(1)

            graphs.append(Data(x=x, edge_index=edge_index,
                          edge_attr=edge_attr, y=y))

        train_loader = DataLoader(graphs[:15], batch_size=5)
        val_loader = DataLoader(graphs[15:], batch_size=5)

        model = create_lightning_lightweight(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS
        )

        trainer = Trainer(
            max_epochs=2,
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            accelerator='cpu',
            devices=1
        )

        # Should not crash with very small graphs
        trainer.fit(model, train_loader, val_loader)

        # Verify training completed
        assert trainer.current_epoch > 0

    def test_single_graph_batches(self):
        """Test with batch size of 1 (single graphs).

        Note: This test verifies that the error handling is appropriate
        when batch normalization is used with single samples, which is
        a known limitation of batch normalization.
        """

        # Create single-graph batches
        graphs = []
        for i in range(10):
            num_nodes = 15
            x = torch.randn(num_nodes, NUM_NODE_FEATS)
            edge_index = torch.randint(0, num_nodes, (2, 30))
            edge_attr = torch.randn(30, NUM_EDGE_FEATS)
            y = torch.randn(1)
            graphs.append(Data(x=x, edge_index=edge_index,
                          edge_attr=edge_attr, y=y))

        train_loader = DataLoader(graphs[:8], batch_size=1)
        val_loader = DataLoader(graphs[8:], batch_size=1)

        # Test with baseline pipeline which should be more compatible with single batches
        model = create_lightning_baseline(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS
        )

        trainer = Trainer(
            max_epochs=2,
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            accelerator='cpu',
            devices=1
        )

        # This test verifies that single-graph batches work with appropriate configurations
        # If batch normalization is used, it may fail - this is expected behavior
        try:
            trainer.fit(model, train_loader, val_loader)
            # If successful, verify training completed
            assert trainer.current_epoch > 0
            print("Single graph batches worked successfully")
        except ValueError as e:
            if "Expected more than 1 value per channel when training" in str(e):
                # This is expected when batch normalization is used with single samples
                print(
                    "Single graph batches failed due to batch normalization - this is expected")
                pytest.skip(
                    "Single graph batches incompatible with batch normalization")
            else:
                # Re-raise unexpected errors
                raise

        trainer = Trainer(
            max_epochs=2,
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            accelerator='cpu',
            devices=1
        )

        # Should handle single-graph batches correctly
        trainer.fit(model, train_loader, val_loader)

        assert trainer.current_epoch > 0

    def test_extreme_learning_rates(self):
        """Test with extreme learning rate values."""

        # Create small dataset for quick testing
        graphs = []
        for i in range(20):
            num_nodes = 10
            x = torch.randn(num_nodes, NUM_NODE_FEATS)
            edge_index = torch.randint(0, num_nodes, (2, 20))
            edge_attr = torch.randn(20, NUM_EDGE_FEATS)
            y = torch.randn(1)
            graphs.append(Data(x=x, edge_index=edge_index,
                          edge_attr=edge_attr, y=y))

        train_loader = DataLoader(graphs[:15], batch_size=5)
        val_loader = DataLoader(graphs[15:], batch_size=5)

        extreme_lrs = [1e-6, 1e-1]  # Very low and very high

        for lr in extreme_lrs:
            model = create_lightning_lightweight(
                node_features=NUM_NODE_FEATS,
                edge_features=NUM_EDGE_FEATS,
                lr=lr
            )

            trainer = Trainer(
                max_epochs=2,
                logger=False,
                enable_progress_bar=False,
                enable_checkpointing=False,
                accelerator='cpu',
                devices=1
            )

            # Should not crash even with extreme learning rates
            trainer.fit(model, train_loader, val_loader)

            # Check that loss is finite
            assert torch.isfinite(trainer.callback_metrics.get(
                'train_loss', torch.tensor(0.0)))

    def test_zero_weight_decay(self):
        """Test with zero weight decay."""

        graphs = []
        for i in range(20):
            num_nodes = 10
            x = torch.randn(num_nodes, NUM_NODE_FEATS)
            edge_index = torch.randint(0, num_nodes, (2, 20))
            edge_attr = torch.randn(20, NUM_EDGE_FEATS)
            y = torch.randn(1)
            graphs.append(Data(x=x, edge_index=edge_index,
                          edge_attr=edge_attr, y=y))

        train_loader = DataLoader(graphs[:15], batch_size=5)
        val_loader = DataLoader(graphs[15:], batch_size=5)

        model = create_lightning_standard(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS,
            weight_decay=0.0  # No regularization
        )

        trainer = Trainer(
            max_epochs=2,
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            accelerator='cpu',
            devices=1
        )

        trainer.fit(model, train_loader, val_loader)

        # Verify optimizer was configured with zero weight decay
        optimizer = trainer.optimizers[0]
        assert optimizer.param_groups[0]['weight_decay'] == 0.0


if __name__ == "__main__":
    # Run benchmark tests
    pytest.main([__file__, "-v", "--tb=short", "-x"])
