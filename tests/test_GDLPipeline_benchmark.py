"""
Benchmark test suite for GDLPipeline class.

This test suite evaluates the performance characteristics of the GDLPipeline
for different configurations and scales, including:
- Memory usage analysis
- Training speed benchmarks
- Inference speed benchmarks
- Scalability tests
- Parameter efficiency analysis
"""

import pytest
import torch
import torch.nn as nn
import time
import psutil
import os
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from typing import Dict, Any, List, Tuple
import gc

from ACAgraphML.Pipeline.Models.GDLPipeline import (
    GDLPipeline,
    GNNConfig,
    PoolingConfig,
    RegressorConfig,
    create_baseline_pipeline,
    create_standard_pipeline,
    create_advanced_pipeline,
    create_lightweight_pipeline
)


# Benchmark constants
NUM_NODE_FEATS = 28
NUM_EDGE_FEATS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class PerformanceTimer:
    """Context manager for timing operations."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.end_time = time.time()

    @property
    def elapsed(self) -> float:
        return self.end_time - self.start_time if self.end_time else 0


class MemoryProfiler:
    """Memory usage profiler."""

    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        result = {
            'ram_mb': memory_info.rss / 1024 / 1024,
            'peak_ram_mb': memory_info.peak_wss / 1024 / 1024 if hasattr(memory_info, 'peak_wss') else 0
        }

        if torch.cuda.is_available():
            result.update({
                'gpu_allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'gpu_reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                'gpu_max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024
            })

        return result

    @staticmethod
    def reset_peak_memory():
        """Reset peak memory tracking."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        gc.collect()


def create_benchmark_data(
    num_graphs: int,
    min_nodes: int = 10,
    max_nodes: int = 50,
    avg_degree: float = 4.0
) -> Batch:
    """Create benchmark data with specified characteristics."""
    graphs = []

    for i in range(num_graphs):
        num_nodes = torch.randint(min_nodes, max_nodes + 1, (1,)).item()

        # Create node features
        x = torch.randn(num_nodes, NUM_NODE_FEATS)

        # Create edges with specified average degree
        num_edges = int(num_nodes * avg_degree)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Remove self-loops and duplicates
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        edge_index = torch.unique(edge_index, dim=1)

        # Create edge features
        edge_attr = torch.randn(edge_index.size(1), NUM_EDGE_FEATS)

        # Create target
        y = torch.randn(1)

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        graphs.append(graph)

    return Batch.from_data_list(graphs)


class TestGDLPipelinePerformance:
    """Performance benchmark tests for GDLPipeline."""

    @pytest.mark.benchmark
    def test_pipeline_creation_speed(self):
        """Benchmark pipeline creation speed for different configurations."""
        configurations = [
            ("Baseline", create_baseline_pipeline),
            ("Standard", create_standard_pipeline),
            ("Advanced", create_advanced_pipeline),
            ("Lightweight", create_lightweight_pipeline)
        ]

        creation_times = {}

        for name, create_func in configurations:
            with PerformanceTimer(f"Create {name} Pipeline") as timer:
                pipeline = create_func(NUM_NODE_FEATS, NUM_EDGE_FEATS)
                pipeline = pipeline.to(DEVICE)

            creation_times[name] = timer.elapsed
            params = pipeline.get_num_parameters()

            print(f"{name} Pipeline:")
            print(f"  Creation time: {timer.elapsed:.4f}s")
            print(f"  Parameters: {params['total']:,}")
            print(
                f"  Parameters/second: {params['total']/max(timer.elapsed, 1e-6):,.0f}")

        # Assert reasonable creation times (should be under 1 second)
        for name, time_taken in creation_times.items():
            assert time_taken < 1.0, f"{name} pipeline creation took too long: {time_taken:.4f}s"

    @pytest.mark.benchmark
    @pytest.mark.parametrize("batch_size", [1, 8, 16, 32, 64])
    def test_forward_pass_scaling(self, batch_size):
        """Benchmark forward pass speed vs batch size."""
        pipeline = create_standard_pipeline(
            NUM_NODE_FEATS, NUM_EDGE_FEATS).to(DEVICE)
        pipeline.eval()

        # Create test data
        batch = create_benchmark_data(
            batch_size, min_nodes=15, max_nodes=25).to(DEVICE)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = pipeline(batch.x, batch.edge_index,
                             batch.edge_attr, batch.batch)

        # Benchmark
        MemoryProfiler.reset_peak_memory()

        with torch.no_grad():
            with PerformanceTimer(f"Forward pass (batch_size={batch_size})") as timer:
                predictions = pipeline(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        memory_usage = MemoryProfiler.get_memory_usage()

        # Calculate performance metrics
        elapsed_time = max(timer.elapsed, 1e-6)  # Prevent division by zero
        graphs_per_second = batch_size / elapsed_time
        nodes_processed = batch.x.size(0)
        nodes_per_second = nodes_processed / elapsed_time

        print(f"Batch size {batch_size}:")
        print(f"  Time: {timer.elapsed:.4f}s")
        print(f"  Graphs/sec: {graphs_per_second:.1f}")
        print(f"  Nodes/sec: {nodes_per_second:.1f}")
        print(
            f"  Memory: {memory_usage.get('gpu_allocated_mb', memory_usage['ram_mb']):.1f} MB")

        # Assertions
        assert predictions.shape[0] == batch_size
        assert timer.elapsed < 5.0, f"Forward pass too slow for batch size {batch_size}"
        assert graphs_per_second > 1.0, f"Throughput too low: {graphs_per_second:.1f} graphs/sec"

    @pytest.mark.benchmark
    @pytest.mark.parametrize("pipeline_config", [
        ("Lightweight", {"gnn_hidden": 32, "gnn_layers": 2}),
        ("Standard", {"gnn_hidden": 128, "gnn_layers": 4}),
        ("Large", {"gnn_hidden": 256, "gnn_layers": 6})
    ])
    def test_model_size_vs_speed(self, pipeline_config):
        """Benchmark speed vs model size trade-offs."""
        config_name, config_params = pipeline_config

        pipeline = GDLPipeline(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS,
            gnn_config=GNNConfig(
                hidden_dim=config_params["gnn_hidden"],
                num_layers=config_params["gnn_layers"],
                layer_name="GINEConv"
            ),
            pooling_config=PoolingConfig(pooling_type='attentional'),
            regressor_config=RegressorConfig(
                regressor_type='mlp',
                hidden_dims=[config_params["gnn_hidden"],
                             config_params["gnn_hidden"] // 2]
            )
        ).to(DEVICE)

        # Get model statistics
        params = pipeline.get_num_parameters()

        # Create test data
        batch = create_benchmark_data(
            16, min_nodes=20, max_nodes=30).to(DEVICE)

        # Benchmark forward pass
        pipeline.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(3):
                _ = pipeline(batch.x, batch.edge_index,
                             batch.edge_attr, batch.batch)

            # Benchmark
            MemoryProfiler.reset_peak_memory()
            with PerformanceTimer(f"Forward pass ({config_name})") as timer:
                predictions = pipeline(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        memory_usage = MemoryProfiler.get_memory_usage()

        # Calculate efficiency metrics
        params_per_mb = params['total'] / \
            memory_usage.get('gpu_allocated_mb', memory_usage['ram_mb'])

        print(f"{config_name} Model:")
        print(f"  Parameters: {params['total']:,}")
        print(f"  Forward time: {timer.elapsed:.4f}s")
        print(
            f"  Memory: {memory_usage.get('gpu_allocated_mb', memory_usage['ram_mb']):.1f} MB")
        print(f"  Params/MB: {params_per_mb:,.0f}")
        print(f"  Throughput: {16/max(timer.elapsed, 1e-6):.1f} graphs/sec")

        # Assertions
        assert torch.isfinite(predictions).all()
        assert timer.elapsed < 10.0, f"{config_name} model too slow"

    @pytest.mark.benchmark
    def test_training_step_performance(self):
        """Benchmark training step performance."""
        pipeline = create_standard_pipeline(
            NUM_NODE_FEATS, NUM_EDGE_FEATS).to(DEVICE)
        optimizer = torch.optim.Adam(pipeline.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Create training data
        batch = create_benchmark_data(
            32, min_nodes=15, max_nodes=25).to(DEVICE)
        targets = torch.randn(32).to(DEVICE)

        # Warmup
        pipeline.train()
        for _ in range(3):
            optimizer.zero_grad()
            predictions = pipeline(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

        # Benchmark training step
        MemoryProfiler.reset_peak_memory()

        with PerformanceTimer("Training step") as timer:
            optimizer.zero_grad()
            predictions = pipeline(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

        memory_usage = MemoryProfiler.get_memory_usage()

        print(f"Training step performance:")
        print(f"  Time: {timer.elapsed:.4f}s")
        print(
            f"  Memory: {memory_usage.get('gpu_max_allocated_mb', memory_usage['ram_mb']):.1f} MB")
        print(f"  Loss: {loss.item():.6f}")

        # Assertions
        assert timer.elapsed < 2.0, "Training step too slow"
        assert torch.isfinite(loss)

    @pytest.mark.benchmark
    @pytest.mark.parametrize("gnn_layer", ["GCN", "SAGE", "GINEConv", "GAT"])
    def test_gnn_architecture_speed_comparison(self, gnn_layer):
        """Compare speed of different GNN architectures."""
        pipeline = GDLPipeline(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS,
            gnn_config=GNNConfig(
                hidden_dim=64,
                num_layers=3,
                layer_name=gnn_layer
            ),
            pooling_config=PoolingConfig(pooling_type='mean'),
            regressor_config=RegressorConfig(regressor_type='linear')
        ).to(DEVICE)

        # Create test data
        batch = create_benchmark_data(
            16, min_nodes=20, max_nodes=30).to(DEVICE)

        # Benchmark
        pipeline.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(3):
                _ = pipeline(batch.x, batch.edge_index,
                             batch.edge_attr, batch.batch)

            # Benchmark
            with PerformanceTimer(f"{gnn_layer} forward pass") as timer:
                predictions = pipeline(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        params = pipeline.get_num_parameters()
        elapsed_time = max(timer.elapsed, 1e-6)  # Prevent division by zero
        throughput = 16 / elapsed_time

        print(f"{gnn_layer} Architecture:")
        print(f"  Time: {timer.elapsed:.4f}s")
        print(f"  Parameters: {params['total']:,}")
        print(f"  Throughput: {throughput:.1f} graphs/sec")

        # Assertions
        assert torch.isfinite(predictions).all()
        assert timer.elapsed < 5.0, f"{gnn_layer} too slow"
        assert throughput > 1.0, f"{gnn_layer} throughput too low"

    @pytest.mark.benchmark
    def test_embedding_extraction_performance(self):
        """Benchmark embedding extraction methods."""
        pipeline = create_standard_pipeline(
            NUM_NODE_FEATS, NUM_EDGE_FEATS).to(DEVICE)
        batch = create_benchmark_data(
            16, min_nodes=20, max_nodes=30).to(DEVICE)

        pipeline.eval()

        extraction_methods = [
            ("Node embeddings", lambda: pipeline.get_node_embeddings(
                batch.x, batch.edge_index, batch.edge_attr)),
            ("Graph embeddings", lambda: pipeline.get_graph_embeddings(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch)),
            ("All embeddings", lambda: pipeline.get_all_embeddings(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch))
        ]

        for method_name, extract_func in extraction_methods:
            # Warmup
            for _ in range(3):
                _ = extract_func()

            # Benchmark
            with PerformanceTimer(method_name) as timer:
                embeddings = extract_func()

            print(f"{method_name}:")
            print(f"  Time: {timer.elapsed:.4f}s")
            if isinstance(embeddings, dict):
                for key, tensor in embeddings.items():
                    print(f"  {key}: {tensor.shape}")
            else:
                print(f"  Shape: {embeddings.shape}")

            assert timer.elapsed < 1.0, f"{method_name} too slow"

    @pytest.mark.benchmark
    @pytest.mark.parametrize("graph_size", [
        ("Small", 10, 20),
        ("Medium", 50, 100),
        ("Large", 100, 200)
    ])
    def test_graph_size_scaling(self, graph_size):
        """Test performance scaling with graph size."""
        size_name, min_nodes, max_nodes = graph_size

        pipeline = create_lightweight_pipeline(
            NUM_NODE_FEATS, NUM_EDGE_FEATS).to(DEVICE)
        batch = create_benchmark_data(
            8, min_nodes=min_nodes, max_nodes=max_nodes).to(DEVICE)

        total_nodes = batch.x.size(0)
        total_edges = batch.edge_index.size(1)

        pipeline.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(3):
                _ = pipeline(batch.x, batch.edge_index,
                             batch.edge_attr, batch.batch)

            # Benchmark
            MemoryProfiler.reset_peak_memory()
            with PerformanceTimer(f"{size_name} graphs") as timer:
                predictions = pipeline(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        memory_usage = MemoryProfiler.get_memory_usage()

        print(f"{size_name} graphs:")
        print(f"  Nodes: {total_nodes}")
        print(f"  Edges: {total_edges}")
        print(f"  Time: {timer.elapsed:.4f}s")
        print(
            f"  Memory: {memory_usage.get('gpu_allocated_mb', memory_usage['ram_mb']):.1f} MB")
        print(f"  Nodes/sec: {total_nodes/max(timer.elapsed, 1e-6):.1f}")

        # Assertions
        assert torch.isfinite(predictions).all()
        # Allow more time for larger graphs
        max_time = 1.0 if size_name == "Small" else 5.0 if size_name == "Medium" else 15.0
        assert timer.elapsed < max_time, f"{size_name} graphs too slow"


class TestGDLPipelineMemoryEfficiency:
    """Memory efficiency tests for GDLPipeline."""

    @pytest.mark.benchmark
    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated forward passes."""
        pipeline = create_standard_pipeline(
            NUM_NODE_FEATS, NUM_EDGE_FEATS).to(DEVICE)

        # Initial memory measurement
        initial_memory = MemoryProfiler.get_memory_usage()

        pipeline.eval()
        with torch.no_grad():
            for i in range(20):
                batch = create_benchmark_data(
                    8, min_nodes=15, max_nodes=25).to(DEVICE)
                predictions = pipeline(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch)

                # Force cleanup
                del batch, predictions
                if i % 5 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()

        # Final memory measurement
        final_memory = MemoryProfiler.get_memory_usage()

        # Calculate memory increase
        memory_key = 'gpu_allocated_mb' if torch.cuda.is_available() else 'ram_mb'
        memory_increase = final_memory[memory_key] - initial_memory[memory_key]

        print(f"Memory leak test:")
        print(f"  Initial memory: {initial_memory[memory_key]:.1f} MB")
        print(f"  Final memory: {final_memory[memory_key]:.1f} MB")
        print(f"  Memory increase: {memory_increase:.1f} MB")

        # Assert no significant memory leak (allow small increases due to caching)
        assert memory_increase < 100, f"Potential memory leak detected: {memory_increase:.1f} MB increase"

    @pytest.mark.benchmark
    def test_memory_vs_batch_size(self):
        """Test memory usage scaling with batch size."""
        pipeline = create_standard_pipeline(
            NUM_NODE_FEATS, NUM_EDGE_FEATS).to(DEVICE)

        batch_sizes = [1, 4, 8, 16, 32]
        memory_usage = []

        pipeline.eval()

        for batch_size in batch_sizes:
            MemoryProfiler.reset_peak_memory()

            batch = create_benchmark_data(
                batch_size, min_nodes=20, max_nodes=30).to(DEVICE)

            with torch.no_grad():
                predictions = pipeline(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            memory = MemoryProfiler.get_memory_usage()
            memory_key = 'gpu_max_allocated_mb' if torch.cuda.is_available() else 'ram_mb'
            memory_usage.append(memory[memory_key])

            print(f"Batch size {batch_size}: {memory[memory_key]:.1f} MB")

        # Check that memory scaling is reasonable (should be roughly linear)
        memory_ratios = [memory_usage[i] / memory_usage[0]
                         for i in range(len(memory_usage))]
        batch_ratios = [batch_sizes[i] / batch_sizes[0]
                        for i in range(len(batch_sizes))]

        # Memory should scale sub-linearly or linearly, not super-linearly
        for i in range(1, len(memory_ratios)):
            assert memory_ratios[i] <= batch_ratios[i] * 2, \
                f"Memory scaling too aggressive: {memory_ratios[i]:.2f}x for {batch_ratios[i]}x batch size"


class TestGDLPipelineScalabilityLimits:
    """Test scalability limits and edge cases."""

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_maximum_graph_size(self):
        """Test performance with very large graphs."""
        pipeline = create_lightweight_pipeline(
            NUM_NODE_FEATS, NUM_EDGE_FEATS).to(DEVICE)

        # Create a single very large graph
        num_nodes = 1000
        x = torch.randn(num_nodes, NUM_NODE_FEATS).to(DEVICE)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 4)).to(DEVICE)
        edge_attr = torch.randn(edge_index.size(1), NUM_EDGE_FEATS).to(DEVICE)

        pipeline.eval()

        try:
            with PerformanceTimer("Large graph forward pass") as timer:
                with torch.no_grad():
                    predictions = pipeline(x, edge_index, edge_attr)

            print(f"Large graph test:")
            print(f"  Nodes: {num_nodes}")
            print(f"  Edges: {edge_index.size(1)}")
            print(f"  Time: {timer.elapsed:.4f}s")
            print(f"  Success: True")

            assert torch.isfinite(predictions).all()
            assert timer.elapsed < 30.0, "Large graph processing too slow"

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(
                    f"Large graph test: Out of memory with {num_nodes} nodes")
                pytest.skip("Insufficient memory for large graph test")
            else:
                raise

    @pytest.mark.benchmark
    def test_configuration_efficiency_comparison(self):
        """Compare efficiency of different pipeline configurations."""
        configurations = {
            "Minimal": {
                'gnn_config': GNNConfig(hidden_dim=32, num_layers=2, layer_name="GCN"),
                'pooling_config': PoolingConfig(pooling_type='mean'),
                'regressor_config': RegressorConfig(regressor_type='linear')
            },
            "Balanced": {
                'gnn_config': GNNConfig(hidden_dim=128, num_layers=4, layer_name="GINEConv"),
                'pooling_config': PoolingConfig(pooling_type='attentional'),
                'regressor_config': RegressorConfig(regressor_type='mlp')
            },
            "High-Performance": {
                'gnn_config': GNNConfig(hidden_dim=256, num_layers=5, layer_name="GAT"),
                'pooling_config': PoolingConfig(pooling_type='set2set'),
                'regressor_config': RegressorConfig(regressor_type='ensemble_mlp')
            }
        }

        batch = create_benchmark_data(
            16, min_nodes=20, max_nodes=30).to(DEVICE)
        results = {}

        for config_name, config in configurations.items():
            pipeline = GDLPipeline(
                node_features=NUM_NODE_FEATS,
                edge_features=NUM_EDGE_FEATS,
                **config
            ).to(DEVICE)

            # Benchmark
            pipeline.eval()
            with torch.no_grad():
                # Warmup
                for _ in range(3):
                    _ = pipeline(batch.x, batch.edge_index,
                                 batch.edge_attr, batch.batch)

                # Timing
                MemoryProfiler.reset_peak_memory()
                with PerformanceTimer(config_name) as timer:
                    predictions = pipeline(
                        batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            memory = MemoryProfiler.get_memory_usage()
            params = pipeline.get_num_parameters()

            # Calculate efficiency metrics
            elapsed_time = max(timer.elapsed, 1e-6)  # Prevent division by zero
            throughput = 16 / elapsed_time
            params_per_mb = params['total'] / \
                memory.get('gpu_allocated_mb', memory['ram_mb'])
            # throughput per million parameters
            efficiency_score = throughput / (params['total'] / 1e6)

            results[config_name] = {
                'time': elapsed_time,
                'throughput': throughput,
                'memory_mb': memory.get('gpu_allocated_mb', memory['ram_mb']),
                'parameters': params['total'],
                'params_per_mb': params_per_mb,
                'efficiency_score': efficiency_score
            }

            print(f"{config_name} Configuration:")
            print(f"  Time: {elapsed_time:.4f}s")
            print(f"  Throughput: {throughput:.1f} graphs/sec")
            print(f"  Memory: {results[config_name]['memory_mb']:.1f} MB")
            print(f"  Parameters: {params['total']:,}")
            print(f"  Efficiency: {efficiency_score:.2f} graphs/sec/Mparam")
            print()

        # Assertions
        for config_name, result in results.items():
            assert result['throughput'] > 0.5, f"{config_name} throughput too low"
            assert torch.isfinite(predictions).all()


if __name__ == "__main__":
    # Run benchmark tests
    pytest.main([__file__ + "::TestGDLPipelinePerformance",
                "-v", "-m", "benchmark"])
