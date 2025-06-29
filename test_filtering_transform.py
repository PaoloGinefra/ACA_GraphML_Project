"""
Test script to demonstrate the FilteringTransform functionality.
"""

import torch
from torch_geometric.data import Data
from src.ACAgraphML.Transforms import FilteringTransform


def create_sample_data(target_value):
    """Create a sample graph with a specific target value."""
    return Data(
        x=torch.randn(5, 3),  # 5 nodes with 3 features each
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),  # Simple chain
        y=torch.tensor([target_value])  # Target value
    )


def test_filtering_transform():
    """Test the FilteringTransform with various scenarios."""

    print("Testing FilteringTransform...")

    # Create sample graphs with different target values
    graphs = [
        create_sample_data(0.5),   # Should pass k=1.0 filter
        create_sample_data(1.5),   # Should be filtered out with k=1.0
        create_sample_data(-0.8),  # Should pass k=1.0 filter (|−0.8| = 0.8)
        # Should be filtered out with k=1.0 (|−2.0| = 2.0)
        create_sample_data(-2.0),
        create_sample_data(1.0),   # Edge case: exactly k=1.0
    ]

    target_values = [0.5, 1.5, -0.8, -2.0, 1.0]

    # Test with k=1.0 (default behavior: keep |y| <= k)
    print("\n1. Testing with k=1.0 (keep |y| <= 1.0):")
    transform = FilteringTransform(k=1.0)
    print(f"Transform: {transform}")

    for i, (graph, target) in enumerate(zip(graphs, target_values)):
        result = transform(graph)
        status = "KEPT" if result is not None else "FILTERED OUT"
        print(
            f"  Graph {i+1}: target={target:4.1f}, |target|={abs(target):4.1f} -> {status}")

    # Test with k=1.0 and invert=True (keep |y| > k)
    print("\n2. Testing with k=1.0, invert=True (keep |y| > 1.0):")
    transform_inverted = FilteringTransform(k=1.0, invert=True)
    print(f"Transform: {transform_inverted}")

    for i, (graph, target) in enumerate(zip(graphs, target_values)):
        result = transform_inverted(graph)
        status = "KEPT" if result is not None else "FILTERED OUT"
        print(
            f"  Graph {i+1}: target={target:4.1f}, |target|={abs(target):4.1f} -> {status}")

    # Test with different k value
    print("\n3. Testing with k=0.5 (keep |y| <= 0.5):")
    transform_strict = FilteringTransform(k=0.5)
    print(f"Transform: {transform_strict}")

    for i, (graph, target) in enumerate(zip(graphs, target_values)):
        result = transform_strict(graph)
        status = "KEPT" if result is not None else "FILTERED OUT"
        print(
            f"  Graph {i+1}: target={target:4.1f}, |target|={abs(target):4.1f} -> {status}")

    # Test with multi-dimensional target
    print("\n4. Testing with multi-dimensional target:")
    multi_target_graph = Data(
        x=torch.randn(3, 2),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        y=torch.tensor([0.5, -1.5, 0.3])  # Max |y| = 1.5
    )

    transform_multi = FilteringTransform(k=1.0)
    result = transform_multi(multi_target_graph)
    status = "KEPT" if result is not None else "FILTERED OUT"
    print(
        f"  Multi-dim target: y={multi_target_graph.y.tolist()}, max|y|=1.5 -> {status}")

    print("\nTesting completed!")


if __name__ == "__main__":
    test_filtering_transform()
