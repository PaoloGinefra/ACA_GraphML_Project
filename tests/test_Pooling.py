"""
Comprehensive test suite for Pooling classes.

This test suite validates all pooling types implemented in the Pooling module
including mean, max, add, attentional, and set2set pooling operations.
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from ACAgraphML.Pipeline.Models.Pooling import Pooling, AttentionalPooling


class TestAttentionalPooling:
    """Test class for AttentionalPooling module."""

    @pytest.fixture(scope="class")
    def setup_attentional_pooling(self):
        """Set up test data for AttentionalPooling."""
        hidden_dim = 64
        pooling_layer = AttentionalPooling(hidden_dim=hidden_dim)

        # Create test data with multiple graphs
        # Graph 1: 3 nodes
        x1 = torch.randn(3, hidden_dim)
        edge_index1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        data1 = Data(x=x1, edge_index=edge_index1)

        # Graph 2: 4 nodes
        x2 = torch.randn(4, hidden_dim)
        edge_index2 = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        data2 = Data(x=x2, edge_index=edge_index2)

        # Create batch
        batch_data = Batch.from_data_list([data1, data2])

        return {
            'pooling_layer': pooling_layer,
            'batch_data': batch_data,
            'hidden_dim': hidden_dim,
            'total_nodes': 7,  # 3 + 4
            'num_graphs': 2
        }

    def test_attentional_pooling_initialization(self, setup_attentional_pooling):
        """Test AttentionalPooling initialization."""
        data = setup_attentional_pooling
        pooling_layer = data['pooling_layer']
        hidden_dim = data['hidden_dim']

        assert isinstance(pooling_layer, AttentionalPooling)
        assert isinstance(pooling_layer.attention_layer, nn.Sequential)
        assert len(pooling_layer.attention_layer) == 3  # Linear, Tanh, Linear

        # Check layer dimensions
        first_layer = pooling_layer.attention_layer[0]
        last_layer = pooling_layer.attention_layer[2]
        assert first_layer.in_features == hidden_dim
        assert first_layer.out_features == hidden_dim
        assert last_layer.in_features == hidden_dim
        assert last_layer.out_features == 1

    def test_attentional_pooling_forward_shape(self, setup_attentional_pooling):
        """Test AttentionalPooling forward pass output shape."""
        data = setup_attentional_pooling
        pooling_layer = data['pooling_layer']
        batch_data = data['batch_data']
        hidden_dim = data['hidden_dim']
        num_graphs = data['num_graphs']

        output = pooling_layer(batch_data.x, batch_data.batch)

        # Check output shape
        assert output.shape == (num_graphs, hidden_dim)
        assert output.dtype == torch.float32

    def test_attentional_pooling_different_dimensions(self):
        """Test AttentionalPooling with different hidden dimensions."""
        hidden_dims = [16, 32, 128, 256]

        for hidden_dim in hidden_dims:
            pooling_layer = AttentionalPooling(hidden_dim=hidden_dim)

            # Create simple test data
            x = torch.randn(5, hidden_dim)
            batch = torch.tensor([0, 0, 1, 1, 1])

            output = pooling_layer(x, batch)

            assert output.shape == (2, hidden_dim)  # 2 graphs

    def test_attentional_pooling_single_graph(self):
        """Test AttentionalPooling with a single graph."""
        hidden_dim = 32
        pooling_layer = AttentionalPooling(hidden_dim=hidden_dim)

        # Single graph with 4 nodes
        x = torch.randn(4, hidden_dim)
        batch = torch.zeros(4, dtype=torch.long)

        output = pooling_layer(x, batch)

        assert output.shape == (1, hidden_dim)

    def test_attentional_pooling_empty_graph(self):
        """Test AttentionalPooling behavior with empty input."""
        hidden_dim = 64
        pooling_layer = AttentionalPooling(hidden_dim=hidden_dim)

        # Empty input
        x = torch.empty(0, hidden_dim)
        batch = torch.empty(0, dtype=torch.long)

        output = pooling_layer(x, batch)

        assert output.shape == (0, hidden_dim)


class TestPooling:
    """Test class for Pooling module with comprehensive coverage of all pooling types."""

    @pytest.fixture(scope="class")
    def setup_pooling_data(self):
        """Set up test data for Pooling tests."""
        hidden_dim = 64

        # Create test data with multiple graphs of different sizes
        # Graph 1: 3 nodes
        x1 = torch.randn(3, hidden_dim)
        edge_index1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        data1 = Data(x=x1, edge_index=edge_index1)

        # Graph 2: 5 nodes
        x2 = torch.randn(5, hidden_dim)
        edge_index2 = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
        data2 = Data(x=x2, edge_index=edge_index2)

        # Graph 3: 2 nodes
        x3 = torch.randn(2, hidden_dim)
        edge_index3 = torch.tensor([[0, 1], [1, 0]])
        data3 = Data(x=x3, edge_index=edge_index3)

        # Create batch
        batch_data = Batch.from_data_list([data1, data2, data3])

        return {
            'batch_data': batch_data,
            'hidden_dim': hidden_dim,
            'total_nodes': 10,  # 3 + 5 + 2
            'num_graphs': 3,
            'graph_sizes': [3, 5, 2]
        }

    @pytest.mark.parametrize("pooling_type,expected_output_dim", [
        ("mean", 64),
        ("max", 64),
        ("add", 64),
        ("attentional", 64),
        ("set2set", 128),  # Set2Set doubles the dimension
    ])
    def test_pooling_initialization(self, pooling_type, expected_output_dim):
        """Test Pooling initialization for all pooling types."""
        hidden_dim = 64
        processing_steps = 3

        pooling_layer = Pooling(
            pooling_type=pooling_type,
            hidden_dim=hidden_dim,
            processing_steps=processing_steps
        )

        assert pooling_layer.pooling_type == pooling_type

        if pooling_type == 'attentional':
            assert isinstance(pooling_layer.pooling_layer, AttentionalPooling)
        elif pooling_type == 'set2set':
            assert pooling_layer.pooling_layer is not None
            # Set2Set layer should be present
            assert hasattr(pooling_layer.pooling_layer, 'in_channels')
        else:
            assert pooling_layer.pooling_layer is None

    @pytest.mark.parametrize("pooling_type", [
        "mean", "max", "add", "attentional", "set2set"
    ])
    def test_pooling_forward_shape(self, setup_pooling_data, pooling_type):
        """Test forward pass shape for all pooling types."""
        data = setup_pooling_data
        batch_data = data['batch_data']
        hidden_dim = data['hidden_dim']
        num_graphs = data['num_graphs']

        pooling_layer = Pooling(
            pooling_type=pooling_type,
            hidden_dim=hidden_dim,
            processing_steps=3
        )

        output = pooling_layer(batch_data.x, batch_data.batch)

        # Check output shape
        expected_out_dim = hidden_dim * 2 if pooling_type == 'set2set' else hidden_dim
        assert output.shape == (num_graphs, expected_out_dim)
        assert output.dtype == torch.float32

    @pytest.mark.parametrize("pooling_type", [
        "mean", "max", "add", "attentional", "set2set"
    ])
    def test_pooling_output_values(self, pooling_type):
        """Test that pooling operations produce reasonable output values."""
        hidden_dim = 32

        # Create controlled test data
        # Graph 1: all ones
        x1 = torch.ones(3, hidden_dim)
        # Graph 2: all twos
        x2 = torch.full((2, hidden_dim), 2.0)

        x = torch.cat([x1, x2], dim=0)
        batch = torch.tensor([0, 0, 0, 1, 1])

        pooling_layer = Pooling(
            pooling_type=pooling_type,
            hidden_dim=hidden_dim,
            processing_steps=2
        )

        output = pooling_layer(x, batch)

        if pooling_type == 'mean':
            # Mean of [1,1,1] should be 1, mean of [2,2] should be 2
            assert torch.allclose(output[0], torch.ones(hidden_dim), atol=1e-6)
            assert torch.allclose(output[1], torch.full(
                (hidden_dim,), 2.0), atol=1e-6)
        elif pooling_type == 'max':
            # Max of [1,1,1] should be 1, max of [2,2] should be 2
            assert torch.allclose(output[0], torch.ones(hidden_dim), atol=1e-6)
            assert torch.allclose(output[1], torch.full(
                (hidden_dim,), 2.0), atol=1e-6)
        elif pooling_type == 'add':
            # Sum of [1,1,1] should be 3, sum of [2,2] should be 4
            assert torch.allclose(output[0], torch.full(
                (hidden_dim,), 3.0), atol=1e-6)
            assert torch.allclose(output[1], torch.full(
                (hidden_dim,), 4.0), atol=1e-6)
        # For attentional and set2set, just check that output is finite
        else:
            assert torch.isfinite(output).all()

    def test_pooling_single_node_graphs(self):
        """Test pooling with single-node graphs."""
        hidden_dim = 32

        # Create three single-node graphs
        x = torch.randn(3, hidden_dim)
        batch = torch.tensor([0, 1, 2])

        for pooling_type in ['mean', 'max', 'add', 'attentional']:
            pooling_layer = Pooling(
                pooling_type=pooling_type,
                hidden_dim=hidden_dim
            )

            output = pooling_layer(x, batch)

            # For single-node graphs, all pooling operations should return the node itself
            if pooling_type in ['mean', 'max']:
                assert torch.allclose(output, x, atol=1e-6)
            elif pooling_type == 'add':
                assert torch.allclose(output, x, atol=1e-6)
            else:  # attentional
                # Should be close to original values
                assert output.shape == x.shape

    def test_pooling_empty_batch(self):
        """Test pooling with empty batch."""
        hidden_dim = 64

        x = torch.empty(0, hidden_dim)
        batch = torch.empty(0, dtype=torch.long)

        for pooling_type in ['mean', 'max', 'add', 'attentional']:
            pooling_layer = Pooling(
                pooling_type=pooling_type,
                hidden_dim=hidden_dim
            )

            output = pooling_layer(x, batch)
            assert output.shape == (0, hidden_dim)

    @pytest.mark.parametrize("hidden_dim", [16, 32, 64, 128, 256])
    def test_pooling_different_dimensions(self, hidden_dim):
        """Test pooling with different hidden dimensions."""
        # Create test data
        x = torch.randn(6, hidden_dim)
        batch = torch.tensor([0, 0, 1, 1, 1, 2])

        for pooling_type in ['mean', 'max', 'add', 'attentional']:
            pooling_layer = Pooling(
                pooling_type=pooling_type,
                hidden_dim=hidden_dim
            )

            output = pooling_layer(x, batch)
            expected_out_dim = hidden_dim
            assert output.shape == (3, expected_out_dim)

    def test_pooling_set2set_processing_steps(self):
        """Test Set2Set pooling with different processing steps."""
        hidden_dim = 32
        x = torch.randn(6, hidden_dim)
        batch = torch.tensor([0, 0, 1, 1, 1, 2])

        for processing_steps in [1, 2, 3, 5]:
            pooling_layer = Pooling(
                pooling_type='set2set',
                hidden_dim=hidden_dim,
                processing_steps=processing_steps
            )

            output = pooling_layer(x, batch)
            # Set2Set doubles the dimension
            assert output.shape == (3, hidden_dim * 2)

    def test_pooling_invalid_type(self):
        """Test that invalid pooling type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown pooling type"):
            pooling_layer = Pooling(pooling_type='invalid')
            x = torch.randn(3, 64)
            batch = torch.tensor([0, 0, 1])
            pooling_layer(x, batch)

    def test_pooling_consistency(self):
        """Test that pooling operations are consistent across multiple runs."""
        hidden_dim = 64
        x = torch.randn(8, hidden_dim)
        batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])

        # Test deterministic pooling types
        for pooling_type in ['mean', 'max', 'add']:
            pooling_layer = Pooling(
                pooling_type=pooling_type, hidden_dim=hidden_dim)

            output1 = pooling_layer(x, batch)
            output2 = pooling_layer(x, batch)

            assert torch.allclose(output1, output2, atol=1e-6)

    def test_pooling_gradient_flow(self):
        """Test that gradients flow through pooling operations."""
        hidden_dim = 32
        x = torch.randn(6, hidden_dim, requires_grad=True)
        batch = torch.tensor([0, 0, 1, 1, 1, 2])

        for pooling_type in ['mean', 'max', 'add', 'attentional']:
            pooling_layer = Pooling(
                pooling_type=pooling_type, hidden_dim=hidden_dim)

            output = pooling_layer(x, batch)
            loss = output.sum()
            loss.backward()

            # Check that gradients exist and are not all zero
            assert x.grad is not None
            assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

            # Reset gradients for next iteration
            x.grad.zero_()

    @pytest.mark.parametrize("batch_size", [1, 5, 10, 20])
    def test_pooling_different_batch_sizes(self, batch_size):
        """Test pooling with different batch sizes."""
        hidden_dim = 64
        nodes_per_graph = 4

        # Create batch with specified number of graphs
        x = torch.randn(batch_size * nodes_per_graph, hidden_dim)
        batch = torch.repeat_interleave(
            torch.arange(batch_size), nodes_per_graph)

        for pooling_type in ['mean', 'max', 'add', 'attentional']:
            pooling_layer = Pooling(
                pooling_type=pooling_type, hidden_dim=hidden_dim)

            output = pooling_layer(x, batch)
            assert output.shape[0] == batch_size
            assert output.shape[1] == hidden_dim

    def test_pooling_edge_cases(self):
        """Test pooling with edge cases."""
        hidden_dim = 32

        # Test with very large values
        x_large = torch.full((4, hidden_dim), 1e6)
        batch_large = torch.tensor([0, 0, 1, 1])

        # Test with very small values
        x_small = torch.full((4, hidden_dim), 1e-6)
        batch_small = torch.tensor([0, 0, 1, 1])

        for pooling_type in ['mean', 'max', 'add']:
            pooling_layer = Pooling(
                pooling_type=pooling_type, hidden_dim=hidden_dim)

            output_large = pooling_layer(x_large, batch_large)
            output_small = pooling_layer(x_small, batch_small)

            # Check that outputs are finite
            assert torch.isfinite(output_large).all()
            assert torch.isfinite(output_small).all()
