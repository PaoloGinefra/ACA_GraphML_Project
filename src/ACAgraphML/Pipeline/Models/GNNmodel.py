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
from typing import Optional, List, Literal, Any

LayerName = Literal[
    "SGConv",
    "GraphConv",
    "GCN",
    "SAGE",
    "GINConv",
    "ChebConv",
    "ARMAConv",
    "TAGConv",
    "GAT",
    "GATv2",
    "TransformerConv",
    "GINEConv",
    "PNA"
]


class GNNModel(nn.Module):
    """
    Flexible GNN model supporting multiple layer types for molecular property prediction.
    See module docstring for details.
    """

    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        c_out: int,
        num_layers: int = 4,
        layer_name: LayerName = "GINEConv",
        dp_rate: float = 0.1,
        edge_dim: Optional[int] = None,
        use_residual: bool = True,
        use_layer_norm: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the GNNModel.
        See class docstring for argument details.

        Args:
            c_in (int): Number of input node features.
            c_hidden (int): Hidden layer dimension.
            c_out (int): Number of output features.
            num_layers (int, optional): Number of GNN layers. Default is 4.
            layer_name (LayerName, optional): Type of GNN layer to use (e.g., 'GINEConv', 'GCN', etc.). Default is 'GINEConv'.
            dp_rate (float, optional): Dropout rate. Default is 0.1.
            edge_dim (Optional[int], optional): Edge feature dimension (required for some layers). Default is None.
            use_residual (bool, optional): Whether to use residual connections. Default is True.
            use_layer_norm (bool, optional): Whether to use layer normalization. Default is True.
            **kwargs: Additional arguments for specific GNN layers.

        Attributes:
            supports_edge_attr (bool): Whether the selected layer supports edge attributes.
            layer_name (LayerName): Name of the GNN layer type.
            use_residual (bool): Whether residual connections are used.
            use_layer_norm (bool): Whether layer normalization is used.
            c_in (int): Input feature dimension.
            c_hidden (int): Hidden feature dimension.
            c_out (int): Output feature dimension.
            input_proj (nn.Linear): Optional input projection layer.
            gnn_layers (nn.ModuleList): List of GNN layers.
            layer_norms (nn.ModuleList): List of layer normalization layers.
            dropouts (nn.ModuleList): List of dropout layers.
            output_proj (nn.Linear): Optional output projection layer.
        """
        super().__init__()

        self.supports_edge_attr: bool = layer_name in [
            "GAT", "GATv2", "GINEConv", "PNA", "TransformerConv"]
        self.layer_name: LayerName = layer_name
        self.use_residual: bool = use_residual
        self.use_layer_norm: bool = use_layer_norm
        self.c_in: int = c_in
        self.c_hidden: int = c_hidden
        self.c_out: int = c_out

        # Input projection to hidden dimension
        if (c_in != c_hidden):
            self.input_proj = nn.Linear(c_in, c_hidden)

        self.gnn_layers: nn.ModuleList = nn.ModuleList()
        self.layer_norms: nn.ModuleList = nn.ModuleList()
        self.dropouts: nn.ModuleList = nn.ModuleList()

        for i in range(num_layers):
            if layer_name == "GINEConv":
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
                layer_kwargs = kwargs.copy()
                if self.supports_edge_attr and edge_dim is not None:
                    layer_kwargs['edge_dim'] = edge_dim
                layer_kwargs.setdefault('heads', 8)
                layer_kwargs.setdefault('concat', False)
                gnn_layer = geom_nn.GATConv(c_hidden, c_hidden, **layer_kwargs)

            elif layer_name == "GATv2":
                layer_kwargs = kwargs.copy()
                if self.supports_edge_attr and edge_dim is not None:
                    layer_kwargs['edge_dim'] = edge_dim
                layer_kwargs.setdefault('heads', 8)
                layer_kwargs.setdefault('concat', False)
                gnn_layer = geom_nn.GATv2Conv(
                    c_hidden, c_hidden, **layer_kwargs)

            elif layer_name == "PNA":
                layer_kwargs = kwargs.copy()
                layer_kwargs.setdefault(
                    'aggregators', ['mean', 'min', 'max', 'std'])
                layer_kwargs.setdefault(
                    'scalers', ['identity', 'amplification', 'attenuation'])
                layer_kwargs.setdefault('deg', torch.tensor(
                    [1.0] * 10))
                if edge_dim is not None:
                    layer_kwargs['edge_dim'] = edge_dim
                gnn_layer = geom_nn.PNAConv(c_hidden, c_hidden, **layer_kwargs)

            elif layer_name == "TransformerConv":
                layer_kwargs = kwargs.copy()
                layer_kwargs.setdefault('heads', 8)
                layer_kwargs.setdefault('concat', False)
                if edge_dim is not None:
                    layer_kwargs['edge_dim'] = edge_dim
                gnn_layer = geom_nn.TransformerConv(
                    c_hidden, c_hidden, **layer_kwargs)

            elif layer_name == "SAGE":
                layer_kwargs = kwargs.copy()
                layer_kwargs.setdefault('aggr', 'mean')
                gnn_layer = geom_nn.SAGEConv(
                    c_hidden, c_hidden, **layer_kwargs)

            elif layer_name == "GraphConv":
                layer_kwargs = kwargs.copy()
                layer_kwargs.setdefault('aggr', 'add')
                gnn_layer = geom_nn.GraphConv(
                    c_hidden, c_hidden, **layer_kwargs)

            elif layer_name == "GCN":
                gnn_layer = geom_nn.GCNConv(c_hidden, c_hidden, **kwargs)

            elif layer_name == "ChebConv":
                layer_kwargs = kwargs.copy()
                layer_kwargs.setdefault('K', 3)
                gnn_layer = geom_nn.ChebConv(
                    c_hidden, c_hidden, **layer_kwargs)

            elif layer_name == "ARMAConv":
                layer_kwargs = kwargs.copy()
                layer_kwargs.setdefault('num_stacks', 1)
                layer_kwargs.setdefault('num_layers', 1)
                gnn_layer = geom_nn.ARMAConv(
                    c_hidden, c_hidden, **layer_kwargs)

            elif layer_name == "SGConv":
                layer_kwargs = kwargs.copy()
                layer_kwargs.setdefault('K', 2)
                gnn_layer = geom_nn.SGConv(c_hidden, c_hidden, **layer_kwargs)

            elif layer_name == "TAGConv":
                layer_kwargs = kwargs.copy()
                layer_kwargs.setdefault('K', 3)
                gnn_layer = geom_nn.TAGConv(c_hidden, c_hidden, **layer_kwargs)

            elif layer_name == "GINConv":
                nn_module = nn.Sequential(
                    nn.Linear(c_hidden, c_hidden * 2),
                    nn.ReLU(),
                    nn.Linear(c_hidden * 2, c_hidden)
                )
                layer_kwargs = kwargs.copy()
                gnn_layer = geom_nn.GINConv(nn_module, **layer_kwargs)

            else:
                gnn_layer = geom_nn.GCNConv(c_hidden, c_hidden, **kwargs)

            self.gnn_layers.append(gnn_layer)

            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(c_hidden))

            self.dropouts.append(nn.Dropout(dp_rate))

        if c_hidden != c_out:
            self.output_proj = nn.Linear(c_hidden, c_out)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the GNN model.

        Args:
            x (torch.Tensor): Node features [num_nodes, c_in].
            edge_index (torch.Tensor): Edge indices [2, num_edges].
            edge_attr (Optional[torch.Tensor]): Edge features [num_edges, edge_dim].

        Returns:
            torch.Tensor: Output node features [num_nodes, c_out].
        """
        if (self.c_in != self.c_hidden):
            x = self.input_proj(x)

        x = x.float()
        edge_index = edge_index.long()

        for i, (gnn_layer, dropout) in enumerate(zip(self.gnn_layers, self.dropouts)):
            x_residual = x

            if isinstance(gnn_layer, geom_nn.MessagePassing):
                if self.supports_edge_attr and edge_attr is not None:
                    x = gnn_layer(x, edge_index, edge_attr=edge_attr)
                else:
                    x = gnn_layer(x, edge_index)
            else:
                x = gnn_layer(x)

            if self.use_layer_norm:
                x = self.layer_norms[i](x)

            if self.use_residual and x.shape == x_residual.shape:
                x = x + x_residual

            x = F.relu(x)
            x = dropout(x)

        if (self.c_hidden != self.c_out):
            x = self.output_proj(x)
        return x
