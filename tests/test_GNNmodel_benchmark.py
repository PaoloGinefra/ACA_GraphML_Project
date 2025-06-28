"""
Performance benchmarking tests for GNNModel on ZINC dataset.
These tests help evaluate and compare different GNN layer types.
"""

import pytest
import torch
import torch.nn as nn
import time
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from ACAgraphML.Dataset import ZINC_Dataset
from ACAgraphML.Transforms import OneHotEncodeFeat
from ACAgraphML.Pipeline.Models.GNNmodel import GNNModel


class TestGNNModelBenchmark:
    """Benchmark tests for comparing different GNN layer performance."""

    @pytest.fixture(scope="class")
    def benchmark_data(self):
        """Setup benchmark data."""
        oneHotTransform = OneHotEncodeFeat(28)

        def transform(data):
            data = oneHotTransform(data)
            data.x = data.x.float()
            data.edge_attr = torch.nn.functional.one_hot(
                data.edge_attr.long(), num_classes=4
            ).float()
            return data

        # Load dataset
        dataset = ZINC_Dataset.SMALL_TRAIN.load(transform=transform)

        # Create benchmark subset
        benchmark_size = min(500, len(dataset))
        benchmark_dataset = dataset[:benchmark_size]

        # Different batch sizes for testing
        loaders = {
            'small': DataLoader(benchmark_dataset, batch_size=16, shuffle=False),
            'medium': DataLoader(benchmark_dataset, batch_size=32, shuffle=False),
            'large': DataLoader(benchmark_dataset, batch_size=64, shuffle=False),
        }

        return {
            'loaders': loaders,
            'dataset_size': benchmark_size,
            'num_node_features': 28,
            'num_edge_features': 4
        }

    def test_layer_comparison_benchmark(self, benchmark_data):
        """Comprehensive benchmark comparing all layer types."""

        # Layer configurations optimized for fair comparison
        layer_configs = {            # Low complexity layers
            "SGConv": {"hidden": 64, "layers": 2, "supports_edges": False, "complexity": "low"},
            "GraphConv": {"hidden": 64, "layers": 2, "supports_edges": False, "complexity": "low"},
            "GCN": {"hidden": 64, "layers": 2, "supports_edges": False, "complexity": "low"},

            # Medium complexity layers
            "SAGE": {"hidden": 64, "layers": 3, "supports_edges": False, "complexity": "medium"},
            "GINConv": {"hidden": 64, "layers": 3, "supports_edges": False, "complexity": "medium"},
            "ChebConv": {"hidden": 64, "layers": 2, "supports_edges": False, "complexity": "medium"},
            "TAGConv": {"hidden": 64, "layers": 2, "supports_edges": False, "complexity": "medium"},

            # Medium-high complexity layers
            "GAT": {"hidden": 32, "layers": 2, "supports_edges": True, "complexity": "medium-high"},
            "GATv2": {"hidden": 32, "layers": 2, "supports_edges": True, "complexity": "medium-high"},
            "TransformerConv": {"hidden": 32, "layers": 2, "supports_edges": True, "complexity": "medium-high"},
            "GINEConv": {"hidden": 64, "layers": 3, "supports_edges": True, "complexity": "medium-high"},

            # High complexity layers
            "PNA": {"hidden": 32, "layers": 2, "supports_edges": True, "complexity": "high"},
        }

        results = {}
        loader = benchmark_data['loaders']['medium']  # Use medium batch size

        print("\n" + "="*80)
        print("GNN LAYER PERFORMANCE BENCHMARK ON ZINC DATASET")
        print("="*80)
        print(
            f"{'Layer':<15} {'Params':<10} {'Inference':<12} {'Memory':<10} {'Complexity':<12}")
        print("-"*80)

        for layer_name, config in layer_configs.items():
            try:
                # Create model
                model = GNNModel(
                    c_in=benchmark_data['num_node_features'],
                    c_hidden=config["hidden"],
                    c_out=32,
                    num_layers=config["layers"],
                    layer_name=layer_name,
                    edge_dim=benchmark_data['num_edge_features'] if config["supports_edges"] else None,
                    dp_rate=0.0,  # No dropout for benchmarking
                    use_residual=True,
                    use_layer_norm=True
                )

                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())

                # Benchmark inference time
                model.eval()
                sample_batch = next(iter(loader))

                # Warmup
                with torch.no_grad():
                    for _ in range(3):
                        if config["supports_edges"]:
                            _ = model(
                                sample_batch.x, sample_batch.edge_index, sample_batch.edge_attr)
                        else:
                            _ = model(sample_batch.x, sample_batch.edge_index)

                # Time measurement
                start_time = time.time()
                with torch.no_grad():
                    for _ in range(10):
                        if config["supports_edges"]:
                            output = model(
                                sample_batch.x, sample_batch.edge_index, sample_batch.edge_attr)
                        else:
                            output = model(
                                sample_batch.x, sample_batch.edge_index)

                end_time = time.time()
                avg_inference_time = (end_time - start_time) / \
                    10 * 1000  # Convert to ms

                # Memory usage (approximate)
                model_size_mb = total_params * 4 / \
                    (1024 * 1024)  # Assuming float32

                # Store results
                results[layer_name] = {
                    'params': total_params,
                    'inference_time_ms': avg_inference_time,
                    'memory_mb': model_size_mb,
                    'complexity': config["complexity"],
                    'supports_edges': config["supports_edges"]
                }

                # Print results
                print(
                    f"{layer_name:<15} {total_params:<10} {avg_inference_time:>8.2f}ms {model_size_mb:>6.2f}MB {config['complexity']:<12}")

                # Cleanup
                del model, output

            except Exception as e:
                print(
                    f"{layer_name:<15} {'FAILED':<10} {'ERROR':<12} {'N/A':<10} {str(e)[:20]}")
                results[layer_name] = {'error': str(e)}

        print("-"*80)

        # Analysis
        successful_results = {k: v for k,
                              v in results.items() if 'error' not in v}

        if successful_results:
            # Find best performers in different categories
            fastest = min(successful_results.items(),
                          key=lambda x: x[1]['inference_time_ms'])
            smallest = min(successful_results.items(),
                           key=lambda x: x[1]['params'])
            most_efficient = min(successful_results.items(),
                                 key=lambda x: x[1]['inference_time_ms'] * x[1]['params'])

            print(f"\nPERFORMANCE ANALYSIS:")
            print(
                f"Fastest inference: {fastest[0]} ({fastest[1]['inference_time_ms']:.2f}ms)")
            print(
                f"Smallest model: {smallest[0]} ({smallest[1]['params']} params)")
            print(f"Most efficient: {most_efficient[0]} (time×params score)")

            # Recommendations for ZINC dataset
            print(f"\nRECOMMENDATIONS FOR ZINC DATASET:")
            print(
                f"• For speed: Use {fastest[0]} or other low-complexity layers")
            print(
                f"• For accuracy: Use GINEConv or PNA (if computational resources allow)")
            print(f"• For balance: Use SAGE or GINConv")
            print(f"• For edge features: Use GINEConv, GAT, or TransformerConv")

        return results

    def test_scaling_with_batch_size(self, benchmark_data):
        """Test how different layers scale with batch size."""

        layer_names = ["GCN", "SAGE", "GINEConv", "GAT"]
        batch_sizes = [16, 32, 64]

        results = {}

        print("\n" + "="*60)
        print("BATCH SIZE SCALING ANALYSIS")
        print("="*60)

        for layer_name in layer_names:
            supports_edges = layer_name in ["GINEConv", "GAT"]

            model = GNNModel(
                c_in=benchmark_data['num_node_features'],
                c_hidden=32,
                c_out=16,
                num_layers=2,
                layer_name=layer_name,
                edge_dim=benchmark_data['num_edge_features'] if supports_edges else None,
                dp_rate=0.0
            )

            model.eval()
            layer_results = {}

            print(f"\n{layer_name}:")
            print(f"{'Batch Size':<12} {'Time (ms)':<12} {'Throughput':<15}")
            print("-" * 40)

            for batch_size in batch_sizes:
                # Use consistent data
                loader = benchmark_data['loaders']['small']

                # Create batch of desired size
                all_data = []
                for batch in loader:
                    all_data.extend(batch.to_data_list())
                    if len(all_data) >= batch_size:
                        break

                from torch_geometric.data import Batch
                test_batch = Batch.from_data_list(all_data[:batch_size])

                # Time measurement
                start_time = time.time()
                with torch.no_grad():
                    for _ in range(5):
                        if supports_edges:
                            _ = model(
                                test_batch.x, test_batch.edge_index, test_batch.edge_attr)
                        else:
                            _ = model(test_batch.x, test_batch.edge_index)

                end_time = time.time()
                avg_time = (end_time - start_time) / 5 * 1000  # ms
                if (avg_time == 0):
                    avg_time = 1e-6
                throughput = batch_size / (avg_time / 1000)  # graphs/second

                layer_results[batch_size] = {
                    'time_ms': avg_time,
                    'throughput': throughput
                }

                print(
                    f"{batch_size:<12} {avg_time:>8.2f}    {throughput:>10.1f} g/s")

            results[layer_name] = layer_results
            del model

        return results

    def test_molecular_property_prediction_benchmark(self, benchmark_data):
        """Benchmark specifically for molecular property prediction task."""

        # Test layers that are particularly good for molecular graphs
        molecular_layers = {
            "GINEConv": "Best for molecular graphs - uses edge features",
            "GINConv": "Strong baseline - graph isomorphism network",
            "GAT": "Attention-based - learns important molecular features",
            "SAGE": "Efficient and scalable",
            "TransformerConv": "Modern attention mechanism"
        }

        print("\n" + "="*70)
        print("MOLECULAR PROPERTY PREDICTION BENCHMARK")
        print("="*70)

        loader = benchmark_data['loaders']['medium']
        results = {}

        for layer_name, description in molecular_layers.items():
            print(f"\nTesting {layer_name}: {description}")

            supports_edges = layer_name in [
                "GINEConv", "GAT", "TransformerConv"]

            # Create molecular property prediction model
            gnn_model = GNNModel(
                c_in=benchmark_data['num_node_features'],
                c_hidden=64,
                c_out=32,
                num_layers=4,  # Deeper for molecular understanding
                layer_name=layer_name,
                edge_dim=benchmark_data['num_edge_features'] if supports_edges else None,
                dp_rate=0.1,
                use_residual=True,
                use_layer_norm=True
            )

            # Graph-level prediction head
            prediction_head = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(16, 1)
            )

            # Test forward pass and timing
            sample_batch = next(iter(loader))

            gnn_model.eval()
            prediction_head.eval()

            start_time = time.time()
            with torch.no_grad():
                if supports_edges:
                    node_embeddings = gnn_model(
                        sample_batch.x, sample_batch.edge_index, sample_batch.edge_attr
                    )
                else:
                    node_embeddings = gnn_model(
                        sample_batch.x, sample_batch.edge_index
                    )

                # Pool to graph level
                graph_embeddings = global_mean_pool(
                    node_embeddings, sample_batch.batch)

                # Predict molecular property
                predictions = prediction_head(graph_embeddings)

            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # ms

            # Calculate model complexity
            total_params = sum(p.numel() for p in gnn_model.parameters()) + \
                sum(p.numel() for p in prediction_head.parameters())

            # Check prediction quality (basic checks)
            num_graphs = sample_batch.batch.max().item() + 1
            mae_from_mean = torch.abs(
                predictions.squeeze() - sample_batch.y[:num_graphs]).mean()

            results[layer_name] = {
                'inference_time_ms': inference_time,
                'total_params': total_params,
                'mae_from_mean': mae_from_mean.item(),
                'supports_edges': supports_edges
            }

            print(f"  Inference time: {inference_time:.2f}ms")
            print(f"  Total parameters: {total_params:,}")
            print(
                f"  Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
            print(f"  MAE from targets: {mae_from_mean:.4f}")

            del gnn_model, prediction_head

        print("\n" + "="*70)
        print("MOLECULAR BENCHMARK SUMMARY:")

        # Sort by inference time
        sorted_by_speed = sorted(
            results.items(), key=lambda x: x[1]['inference_time_ms'])
        print(f"\nFastest to slowest:")
        for i, (layer_name, metrics) in enumerate(sorted_by_speed, 1):
            edge_support = "✓" if metrics['supports_edges'] else "✗"
            print(
                f"{i}. {layer_name:<15} {metrics['inference_time_ms']:>6.2f}ms  (Edges: {edge_support})")

        print(f"\nRecommendations for ZINC molecular property prediction:")
        print(f"• Best overall: GINEConv (designed for molecular graphs)")
        print(f"• Fastest: {sorted_by_speed[0][0]}")
        print(
            f"• With edge features: {[k for k, v in results.items() if v['supports_edges']]}")

        return results


def run_quick_benchmark():
    """Quick benchmark that can be run manually."""
    print("Running Quick GNN Benchmark on ZINC Dataset...")

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
    small_dataset = dataset[:50]  # Very small for quick test
    loader = DataLoader(small_dataset, batch_size=16, shuffle=False)

    # Test a few key layers
    test_layers = ["GCN", "SAGE", "GINEConv", "GAT"]

    print(
        f"\nTesting {len(test_layers)} layer types on {len(small_dataset)} molecules...")
    print("-" * 50)

    for layer_name in test_layers:
        try:
            supports_edges = layer_name in ["GINEConv", "GAT"]

            model = GNNModel(
                c_in=28,
                c_hidden=32,
                c_out=16,
                num_layers=2,
                layer_name=layer_name,
                edge_dim=4 if supports_edges else None,
                dp_rate=0.0
            )

            # Quick forward pass
            sample_batch = next(iter(loader))
            model.eval()

            start_time = time.time()
            with torch.no_grad():
                if supports_edges:
                    output = model(
                        sample_batch.x, sample_batch.edge_index, sample_batch.edge_attr)
                else:
                    output = model(sample_batch.x, sample_batch.edge_index)

            inference_time = (time.time() - start_time) * 1000
            params = sum(p.numel() for p in model.parameters())

            print(
                f"{layer_name:<12} {inference_time:>6.2f}ms  {params:>8,} params  ✓")

        except Exception as e:
            print(f"{layer_name:<12} FAILED: {str(e)[:30]}...")

    print("\nQuick benchmark completed!")


if __name__ == "__main__":
    run_quick_benchmark()
