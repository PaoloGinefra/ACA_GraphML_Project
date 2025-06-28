"""
GNN Model for Molecular Property Prediction on ZINC Dataset

This module implements a flexible GNN architecture supporting multiple layer types
optimized for molecular regression tasks. Each layer type offers different complexity
levels and capabilities for capturing molecular features.

Available Layer Types (ordered by complexity):

LOW COMPLEXITY:
- SGConv: Simplified GCN, removes nonlinearities, fast baseline
- GraphConv: Basic message passing, good starting point
- GCN: Classic spectral approach, simple and interpretable

MEDIUM COMPLEXITY:
- SAGE: Sampling-based aggregation, scales well with molecular size
- GINConv: Theoretically powerful for molecular structure distinction
- ChebConv: Spectral CNN with Chebyshev polynomials
- ARMAConv: Autoregressive filters for smooth molecular surfaces
- TAGConv: Topology-adaptive, good for diverse molecular structures

MEDIUM-HIGH COMPLEXITY:
- GAT: Attention-based, learns neighbor importance
- GATv2: Improved attention mechanism
- TransformerConv: Self-attention for long-range dependencies
- GINEConv: GIN with edge features, excellent for molecular graphs

HIGH COMPLEXITY:
- PNA: Principal Neighbourhood Aggregation, very expressive
  Combines multiple aggregation functions for diverse molecular patterns

For ZINC dataset regression:
- GINEConv/GINConv: Recommended for best performance
- GAT/GATv2: Good for complex molecular interactions
- PNA: When computational resources allow, highest expressivity
- SAGE: Good balance of performance and efficiency

Edge Feature Support:
- Layers that use edge attributes: GAT, GATv2, GINEConv, PNA, TransformerConv
- Layers without edge support: All others (GCN, SAGE, GINConv, etc.)
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn


class GNNModel(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        c_out,
        num_layers=4,
        layer_name="GINEConv",
        dp_rate=0.1,
        edge_dim=None,
        use_residual=True,
        use_layer_norm=True,
        **kwargs,
    ):
        super().__init__()

        self.supports_edge_attr = layer_name in [
            "GAT", "GATv2", "GINEConv", "PNA", "TransformerConv"]
        self.layer_name = layer_name
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.c_out = c_out

        # Input projection to hidden dimension
        if (c_in != c_hidden):
            # If input dimension is different from hidden dimension, add a projection layer
            self.input_proj = nn.Linear(c_in, c_hidden)

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(num_layers):
            # Create GNN layer
            if layer_name == "GINEConv":
                # GIN with Edge features - Excellent for molecular graphs
                # Combines node and edge information effectively, proven strong on ZINC
                nn_module = nn.Sequential(
                    nn.Linear(c_hidden, c_hidden * 2),
                    nn.ReLU(),
                    nn.Linear(c_hidden * 2, c_hidden)
                )
                layer_kwargs = kwargs.copy()
                if edge_dim is not None:
                    layer_kwargs['edge_dim'] = edge_dim
                gnn_layer = geom_nn.GINEConv(nn_module, **layer_kwargs)

            elif layer_name == "GAT":
                # Graph Attention Networks - Good for selective information flow
                # Learns importance of different neighbors, useful for complex molecular structures
                layer_kwargs = kwargs.copy()
                if self.supports_edge_attr and edge_dim is not None:
                    layer_kwargs['edge_dim'] = edge_dim
                layer_kwargs.setdefault('heads', 8)
                layer_kwargs.setdefault('concat', False)
                gnn_layer = geom_nn.GATConv(c_hidden, c_hidden, **layer_kwargs)

            elif layer_name == "GATv2":
                # Improved GAT with better expressivity
                # Better attention mechanism, good for capturing complex molecular interactions
                layer_kwargs = kwargs.copy()
                if self.supports_edge_attr and edge_dim is not None:
                    layer_kwargs['edge_dim'] = edge_dim
                layer_kwargs.setdefault('heads', 8)
                layer_kwargs.setdefault('concat', False)
                gnn_layer = geom_nn.GATv2Conv(
                    c_hidden, c_hidden, **layer_kwargs)

            elif layer_name == "PNA":
                # Principal Neighbourhood Aggregation - High complexity, very expressive
                # Combines multiple aggregation functions, excellent for diverse molecular patterns
                layer_kwargs = kwargs.copy()
                # PNA requires aggregators and scalers
                layer_kwargs.setdefault(
                    'aggregators', ['mean', 'min', 'max', 'std'])
                layer_kwargs.setdefault(
                    'scalers', ['identity', 'amplification', 'attenuation'])
                layer_kwargs.setdefault('deg', torch.tensor(
                    [1.0] * 10))  # Degree histogram placeholder
                if edge_dim is not None:
                    layer_kwargs['edge_dim'] = edge_dim
                gnn_layer = geom_nn.PNAConv(c_hidden, c_hidden, **layer_kwargs)

            elif layer_name == "TransformerConv":
                # Graph Transformer - Medium-high complexity
                # Self-attention mechanism, good for long-range dependencies in molecules
                layer_kwargs = kwargs.copy()
                layer_kwargs.setdefault('heads', 8)
                layer_kwargs.setdefault('concat', False)
                if edge_dim is not None:
                    layer_kwargs['edge_dim'] = edge_dim
                gnn_layer = geom_nn.TransformerConv(
                    c_hidden, c_hidden, **layer_kwargs)

            elif layer_name == "SAGE":
                # GraphSAGE - Medium complexity, good scalability
                # Samples and aggregates from neighborhoods, robust for varying molecular sizes
                layer_kwargs = kwargs.copy()
                layer_kwargs.setdefault('aggr', 'mean')
                gnn_layer = geom_nn.SAGEConv(
                    c_hidden, c_hidden, **layer_kwargs)

            elif layer_name == "GraphConv":
                # Simple Graph Convolution - Low complexity baseline
                # Basic message passing, good starting point for molecular feature extraction
                layer_kwargs = kwargs.copy()
                layer_kwargs.setdefault('aggr', 'add')
                gnn_layer = geom_nn.GraphConv(
                    c_hidden, c_hidden, **layer_kwargs)

            elif layer_name == "GCN":
                # Graph Convolutional Network - Low-medium complexity
                # Classic spectral approach, simple and interpretable
                gnn_layer = geom_nn.GCNConv(c_hidden, c_hidden, **kwargs)

            elif layer_name == "ChebConv":
                # Chebyshev Spectral CNN - Medium complexity
                # Uses Chebyshev polynomials, good for capturing spectral properties
                layer_kwargs = kwargs.copy()
                layer_kwargs.setdefault('K', 3)  # Chebyshev polynomial order
                gnn_layer = geom_nn.ChebConv(
                    c_hidden, c_hidden, **layer_kwargs)

            elif layer_name == "ARMAConv":
                # ARMA filters - Medium complexity
                # Autoregressive moving average filters, good for smooth molecular surfaces
                layer_kwargs = kwargs.copy()
                layer_kwargs.setdefault('num_stacks', 1)
                layer_kwargs.setdefault('num_layers', 1)
                gnn_layer = geom_nn.ARMAConv(
                    c_hidden, c_hidden, **layer_kwargs)

            elif layer_name == "SGConv":
                # Simplified Graph Convolution - Low complexity
                # Removes nonlinearities between layers, fast and simple
                layer_kwargs = kwargs.copy()
                layer_kwargs.setdefault('K', 2)  # Number of hops
                gnn_layer = geom_nn.SGConv(c_hidden, c_hidden, **layer_kwargs)

            elif layer_name == "TAGConv":
                # Topology Adaptive GCN - Medium complexity
                # Adapts to local graph topology, good for diverse molecular structures
                layer_kwargs = kwargs.copy()
                layer_kwargs.setdefault('K', 3)  # Number of hops
                gnn_layer = geom_nn.TAGConv(c_hidden, c_hidden, **layer_kwargs)

            elif layer_name == "GINConv":
                # Graph Isomorphism Network - Medium complexity
                # Theoretically powerful for distinguishing molecular structures
                nn_module = nn.Sequential(
                    nn.Linear(c_hidden, c_hidden * 2),
                    nn.ReLU(),
                    nn.Linear(c_hidden * 2, c_hidden)
                )
                layer_kwargs = kwargs.copy()
                gnn_layer = geom_nn.GINConv(nn_module, **layer_kwargs)

            else:
                # Default to GCN
                gnn_layer = geom_nn.GCNConv(c_hidden, c_hidden, **kwargs)

            self.gnn_layers.append(gnn_layer)

            # Layer normalization
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(c_hidden))

            # Dropout
            self.dropouts.append(nn.Dropout(dp_rate))

        # Output projection
        if c_hidden != c_out:
            # If hidden dimension is different from output dimension, add a projection layer
            self.output_proj = nn.Linear(c_hidden, c_out)

    def forward(self, x, edge_index, edge_attr=None):
        # Input projection
        if (self.c_in != self.c_hidden):
            # If input dimension is different from hidden dimension, apply projection
            x = self.input_proj(x)

        x = x.float()
        edge_index = edge_index.long()

        # GNN layers with residual connections
        for i, (gnn_layer, dropout) in enumerate(zip(self.gnn_layers, self.dropouts)):
            x_residual = x

            # Apply GNN layer
            if isinstance(gnn_layer, geom_nn.MessagePassing):
                if self.supports_edge_attr and edge_attr is not None:
                    x = gnn_layer(x, edge_index, edge_attr=edge_attr)
                else:
                    x = gnn_layer(x, edge_index)
            else:
                x = gnn_layer(x)

            # Layer normalization
            if self.use_layer_norm:
                x = self.layer_norms[i](x)

            # Residual connection
            if self.use_residual and x.shape == x_residual.shape:
                x = x + x_residual

            # Activation and dropout
            x = F.relu(x)
            x = dropout(x)

        # Output projection
        if (self.c_hidden != self.c_out):
            # If hidden dimension is different from output dimension, apply projection
            x = self.output_proj(x)
        return x
