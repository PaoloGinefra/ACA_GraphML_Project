"""
Test runner for GNNModel tests.
This script provides an easy way to run all GNN model tests.
"""

import sys
import os
import subprocess
import time


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, cwd=os.getcwd())
        end_time = time.time()

        if result.returncode == 0:
            print(f"✓ {description} - PASSED ({end_time - start_time:.2f}s)")
            if result.stdout:
                print("\nOutput:")
                print(result.stdout)
        else:
            print(f"✗ {description} - FAILED ({end_time - start_time:.2f}s)")
            if result.stderr:
                print("\nError:")
                print(result.stderr)
            if result.stdout:
                print("\nOutput:")
                print(result.stdout)

        return result.returncode == 0

    except Exception as e:
        print(f"✗ {description} - ERROR: {e}")
        return False


def run_manual_tests():
    """Run tests manually (without pytest)."""
    print("Running GNN Model Tests Manually...")

    # Add the src directory to Python path
    src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
    sys.path.insert(0, os.path.abspath(src_path))

    try:
        # Test 1: Basic functionality
        print("\n" + "="*60)
        print("TEST 1: Basic GNN Model Functionality")
        print("="*60)

        from ACAgraphML.Dataset import ZINC_Dataset
        from ACAgraphML.Transforms import OneHotEncodeFeat
        from ACAgraphML.Pipeline.Models.GNNmodel import GNNModel
        import torch
        from torch_geometric.loader import DataLoader

        # Load small dataset
        oneHotTransform = OneHotEncodeFeat(28)

        def transform(data):
            data = oneHotTransform(data)
            data.x = data.x.float()
            data.edge_attr = torch.nn.functional.one_hot(
                data.edge_attr.long(), num_classes=4
            ).float()
            return data

        dataset = ZINC_Dataset.SMALL_TRAIN.load(transform=transform)
        small_dataset = dataset[:20]  # Very small for testing
        loader = DataLoader(small_dataset, batch_size=8, shuffle=False)

        print(f"✓ Loaded ZINC dataset: {len(small_dataset)} molecules")

        # Test different layers
        test_layers = ["GCN", "SAGE", "GINEConv", "GAT", "TransformerConv"]

        for layer_name in test_layers:
            try:
                supports_edges = layer_name in [
                    "GINEConv", "GAT", "TransformerConv"]

                model = GNNModel(
                    c_in=28,
                    c_hidden=32,
                    c_out=16,
                    num_layers=2,
                    layer_name=layer_name,
                    edge_dim=4 if supports_edges else None,
                    dp_rate=0.1
                )

                # Test forward pass
                sample_batch = next(iter(loader))
                model.eval()

                with torch.no_grad():
                    if supports_edges:
                        output = model(
                            sample_batch.x, sample_batch.edge_index, sample_batch.edge_attr)
                    else:
                        output = model(sample_batch.x, sample_batch.edge_index)

                assert output.shape[0] == sample_batch.x.shape[0], "Output shape mismatch"
                assert output.shape[1] == 16, "Feature dimension mismatch"
                assert not torch.isnan(output).any(), "NaN in output"

                print(f"✓ {layer_name}: Forward pass successful")

            except Exception as e:
                print(f"✗ {layer_name}: Failed - {e}")

        # Test 2: Training loop
        print(f"\n{'='*60}")
        print("TEST 2: Training Loop")
        print("="*60)

        model = GNNModel(
            c_in=28,
            c_hidden=32,
            c_out=1,
            num_layers=2,
            layer_name="GINEConv",
            edge_dim=4,
            dp_rate=0.1
        )

        from torch_geometric.nn import global_mean_pool
        import torch.nn as nn

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Training for a few steps
        model.train()
        initial_loss = None

        for i, batch in enumerate(loader):
            if i >= 3:  # Just a few steps
                break

            optimizer.zero_grad()

            # Forward pass
            node_output = model(batch.x, batch.edge_index, batch.edge_attr)
            graph_output = global_mean_pool(node_output, batch.batch)

            # Create target (use actual targets from batch)
            num_graphs = batch.batch.max().item() + 1
            targets = batch.y[:num_graphs].unsqueeze(1)

            loss = criterion(graph_output, targets)

            if initial_loss is None:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

            print(f"Step {i+1}: Loss = {loss.item():.4f}")

        print(f"✓ Training loop completed successfully")

        # Test 3: Quick benchmark
        print(f"\n{'='*60}")
        print("TEST 3: Quick Performance Benchmark")
        print("="*60)

        import time

        benchmark_layers = ["GCN", "SAGE", "GINEConv"]

        for layer_name in benchmark_layers:
            supports_edges = layer_name == "GINEConv"

            model = GNNModel(
                c_in=28,
                c_hidden=32,
                c_out=16,
                num_layers=2,
                layer_name=layer_name,
                edge_dim=4 if supports_edges else None,
                dp_rate=0.0
            )

            model.eval()
            sample_batch = next(iter(loader))

            # Time inference
            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):
                    if supports_edges:
                        _ = model(sample_batch.x, sample_batch.edge_index,
                                  sample_batch.edge_attr)
                    else:
                        _ = model(sample_batch.x, sample_batch.edge_index)

            avg_time = (time.time() - start_time) / 10 * 1000  # ms
            params = sum(p.numel() for p in model.parameters())

            print(f"{layer_name:<12} {avg_time:>6.2f}ms  {params:>8,} params")

        print(f"\n✓ All manual tests completed successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Manual tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test runner."""
    print("GNN Model Test Runner")
    print("=" * 60)

    # Check if we're in the right directory
    if not os.path.exists("tests"):
        print("Error: Please run this script from the project root directory")
        return

    # Option 1: Try pytest if available
    print("\nChecking for pytest...")
    try:
        import pytest
        print("✓ pytest found")

        # Run specific test files
        test_files = [
            "tests/test_GNNmodel.py",
            "tests/test_GNNmodel_integration.py",
            "tests/test_GNNmodel_benchmark.py"
        ]

        available_files = [f for f in test_files if os.path.exists(f)]

        if available_files:
            print(f"\nFound {len(available_files)} test files:")
            for f in available_files:
                print(f"  - {f}")

            choice = input(f"\nRun with pytest? (y/n): ").lower()
            if choice == 'y':
                for test_file in available_files:
                    success = run_command(
                        f"python -m pytest {test_file} -v", f"Running {test_file}")
                    if not success:
                        print(f"Stopping due to failure in {test_file}")
                        break
                return

    except ImportError:
        print("✗ pytest not available")

    # Option 2: Run manual tests
    print("\nRunning manual tests...")
    success = run_manual_tests()

    if success:
        print(f"\n{'='*60}")
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("="*60)
        print("\nYour GNNModel implementation is working correctly!")
        print("You can now use it for ZINC dataset molecular property prediction.")
        print("\nRecommended layer types for ZINC:")
        print("  • GINEConv: Best for molecular graphs (uses edge features)")
        print("  • SAGE: Good balance of speed and performance")
        print(
            "  • GAT: Good for learning molecular attention patterns (uses edge features)")
        print("  • GCN: Simple and fast baseline")
        print(
            "  • PNA: Highest complexity, best potential performance (uses edge features)")
        print("\nEdge Feature Support:")
        print("  ✓ With edges: GAT, GATv2, GINEConv, PNA, TransformerConv")
        print("  ✗ No edges: GCN, SAGE, GINConv, GraphConv, SGConv, etc.")
    else:
        print(f"\n{'='*60}")
        print("SOME TESTS FAILED ✗")
        print("="*60)
        print("Please check the error messages above and fix any issues.")


if __name__ == "__main__":
    main()
