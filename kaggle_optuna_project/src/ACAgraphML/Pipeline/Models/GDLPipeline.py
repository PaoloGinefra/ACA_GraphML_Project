"""
Graph Deep Learning Pipeline for Molecular Property Prediction on ZINC Dataset

This module implements a comprehensive, fully parameterized pipeline that combines:
1. GNN layers for node-level feature extraction
2. Pooling mechanisms for graph-level representation
3. Regression heads for final property prediction

The pipeline is designed for extensive hyperparameter tuning and provides methods
to extract intermediate embeddings at both node and graph levels.

Key Features:
- Fully parameterized for hyperparameter optimization
- Support for multiple GNN architectures (GINEConv, GAT, PNA, etc.)
- Various pooling strategies (mean, max, attention, set2set)
- Multiple regressor types (linear, MLP, residual, ensemble)
- Embedding extraction at different pipeline stages
- Comprehensive logging and monitoring capabilities
- Built-in regularization and normalization options

Usage:
    pipeline = GDLPipeline(
        node_features=28,
        edge_features=4,
        gnn_config={...},
        pooling_config={...},
        regressor_config={...}
    )
    
    # Training/inference
    predictions = pipeline(x, edge_index, edge_attr, batch)
    
    # Extract embeddings
    node_embeddings = pipeline.get_node_embeddings(x, edge_index, edge_attr)
    graph_embeddings = pipeline.get_graph_embeddings(x, edge_index, edge_attr, batch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Literal, List, Union
import warnings
from dataclasses import dataclass

from .GNNmodel import GNNModel, LayerName
from .Pooling import Pooling
from .Regressor import Regressor


@dataclass
class GNNConfig:
    """Configuration for GNN component."""
    hidden_dim: int = 128
    output_dim: Optional[int] = None  # If None, uses hidden_dim
    num_layers: int = 4
    layer_name: LayerName = "GINEConv"
    dropout_rate: float = 0.1
    linear_dropout_rate: float = 0.3
    use_residual: bool = True
    use_layer_norm: bool = True
    # Layer-specific parameters
    gat_heads: int = 8
    gat_concat: bool = False
    pna_aggregators: List[str] = None
    pna_scalers: List[str] = None
    pna_deg: Optional[torch.Tensor] = None
    transformer_heads: int = 8
    sage_aggr: str = 'mean'
    cheb_k: int = 3
    arma_num_stacks: int = 1
    arma_num_layers: int = 1
    sgconv_k: int = 2
    tagconv_k: int = 3

    def __post_init__(self):
        if self.output_dim is None:
            self.output_dim = self.hidden_dim
        if self.pna_aggregators is None:
            self.pna_aggregators = ['mean', 'min', 'max', 'std']
        if self.pna_scalers is None:
            self.pna_scalers = ['identity', 'amplification', 'attenuation']


@dataclass
class PoolingConfig:
    """Configuration for Pooling component."""
    pooling_type: Literal['mean', 'max', 'add',
                          'attentional', 'set2set'] = 'mean'
    processing_steps: int = 3  # For set2set pooling
    attention_hidden_multiplier: float = 1.0  # For attentional pooling


@dataclass
class RegressorConfig:
    """Configuration for Regressor component."""
    regressor_type: Literal['linear', 'mlp', 'residual_mlp',
                            'attention_mlp', 'ensemble_mlp'] = 'mlp'
    # Linear options
    linear_dropout: float = 0.0
    # MLP options
    hidden_dims: List[int] = None
    mlp_dropout: float = 0.15
    normalization: Literal['none', 'batch', 'layer'] = 'batch'
    activation: str = 'relu'
    # Residual MLP options
    residual_hidden_dim: int = 128
    residual_num_layers: int = 3
    residual_dropout: float = 0.1
    residual_normalization: Literal['batch', 'layer'] = 'layer'
    # Attention MLP options
    attention_hidden_dim: int = 128
    attention_num_heads: int = 4
    attention_num_layers: int = 2
    attention_dropout: float = 0.1
    # Ensemble options
    ensemble_num_heads: int = 3
    ensemble_aggregation: Literal['mean', 'weighted'] = 'weighted'

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]


class GDLPipeline(nn.Module):
    """
    Graph Deep Learning Pipeline for molecular property prediction.

    This class combines GNN, Pooling, and Regressor components into a unified
    pipeline with comprehensive parameterization for hyperparameter tuning.
    """

    def __init__(
        self,
        node_features: int,
        edge_features: Optional[int] = None,
        gnn_config: Optional[Union[GNNConfig, Dict[str, Any]]] = None,
        pooling_config: Optional[Union[PoolingConfig, Dict[str, Any]]] = None,
        regressor_config: Optional[Union[RegressorConfig,
                                         Dict[str, Any]]] = None,
        global_dropout: float = 0.0,
        use_batch_norm: bool = False,
        target_mean: Optional[float] = None,
        target_std: Optional[float] = None,
        gradient_clipping: Optional[float] = None,
        freeze_gnn: bool = False,
        freeze_pooling: bool = False,
        **kwargs
    ):
        """
        Initialize the Graph Deep Learning Pipeline.

        Args:
            node_features: Number of input node features
            edge_features: Number of edge features (None if not using edge features)
            gnn_config: Configuration for GNN component
            pooling_config: Configuration for Pooling component  
            regressor_config: Configuration for Regressor component
            global_dropout: Additional dropout applied after graph embeddings
            use_batch_norm: Whether to apply batch normalization after graph embeddings
            target_mean: Mean of target values for normalization
            target_std: Standard deviation of target values for normalization
            gradient_clipping: Maximum gradient norm (None to disable)
            freeze_gnn: Whether to freeze GNN parameters
            freeze_pooling: Whether to freeze pooling parameters
            **kwargs: Additional arguments passed to individual components
        """
        super().__init__()

        # Convert dict configs to dataclass configs
        if isinstance(gnn_config, dict):
            gnn_config = GNNConfig(**gnn_config)
        elif gnn_config is None:
            gnn_config = GNNConfig()

        if isinstance(pooling_config, dict):
            pooling_config = PoolingConfig(**pooling_config)
        elif pooling_config is None:
            pooling_config = PoolingConfig()

        if isinstance(regressor_config, dict):
            regressor_config = RegressorConfig(**regressor_config)
        elif regressor_config is None:
            regressor_config = RegressorConfig()

        self.node_features = node_features
        self.edge_features = edge_features
        self.gnn_config = gnn_config
        self.pooling_config = pooling_config
        self.regressor_config = regressor_config
        self.global_dropout = global_dropout
        self.use_batch_norm = use_batch_norm
        self.gradient_clipping = gradient_clipping

        # Target normalization parameters
        self.register_buffer('target_mean', torch.tensor(
            target_mean) if target_mean is not None else None)
        self.register_buffer('target_std', torch.tensor(
            target_std) if target_std is not None else None)

        # Build GNN component
        gnn_kwargs = self._extract_gnn_kwargs(kwargs)
        self.gnn = GNNModel(
            c_in=node_features,
            c_hidden=gnn_config.hidden_dim,
            c_out=gnn_config.output_dim,
            num_layers=gnn_config.num_layers,
            layer_name=gnn_config.layer_name,
            dp_rate=gnn_config.dropout_rate,
            edge_dim=edge_features,
            use_residual=gnn_config.use_residual,
            use_layer_norm=gnn_config.use_layer_norm,
            **gnn_kwargs
        )

        # Calculate pooling input dimension
        pooling_input_dim = gnn_config.output_dim

        # Build Pooling component
        self.pooling = Pooling(
            pooling_type=pooling_config.pooling_type,
            hidden_dim=pooling_input_dim,
            processing_steps=pooling_config.processing_steps
        )

        # Calculate regressor input dimension (set2set doubles the dimension)
        regressor_input_dim = pooling_input_dim
        if pooling_config.pooling_type == 'set2set':
            regressor_input_dim = pooling_input_dim * 2

        # Additional processing layers
        self.graph_dropout = nn.Dropout(
            global_dropout) if global_dropout > 0 else nn.Identity()
        self.graph_batch_norm = nn.BatchNorm1d(
            regressor_input_dim) if use_batch_norm else nn.Identity()

        # Build Regressor component
        regressor_kwargs = self._extract_regressor_kwargs(kwargs)
        self.regressor = Regressor(
            input_dim=regressor_input_dim,
            regressor_type=regressor_config.regressor_type,
            linear_dropout=regressor_config.linear_dropout,
            hidden_dims=regressor_config.hidden_dims,
            mlp_dropout=regressor_config.mlp_dropout,
            normalization=regressor_config.normalization,
            activation=regressor_config.activation,
            residual_hidden_dim=regressor_config.residual_hidden_dim,
            residual_num_layers=regressor_config.residual_num_layers,
            residual_dropout=regressor_config.residual_dropout,
            attention_hidden_dim=regressor_config.attention_hidden_dim,
            attention_num_heads=regressor_config.attention_num_heads,
            attention_num_layers=regressor_config.attention_num_layers,
            attention_dropout=regressor_config.attention_dropout,
            ensemble_num_heads=regressor_config.ensemble_num_heads,
            ensemble_aggregation=regressor_config.ensemble_aggregation,
            **regressor_kwargs
        )

        # Freeze components if requested
        if freeze_gnn:
            for param in self.gnn.parameters():
                param.requires_grad = False
        if freeze_pooling:
            for param in self.pooling.parameters():
                param.requires_grad = False

        # Initialize weights
        self.apply(self._init_weights)

    def _extract_gnn_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract GNN-specific keyword arguments."""
        gnn_kwargs = {}

        # GAT parameters
        if self.gnn_config.layer_name in ["GAT", "GATv2"]:
            gnn_kwargs['heads'] = self.gnn_config.gat_heads
            gnn_kwargs['concat'] = self.gnn_config.gat_concat

        # PNA parameters
        elif self.gnn_config.layer_name == "PNA":
            gnn_kwargs['aggregators'] = self.gnn_config.pna_aggregators
            gnn_kwargs['scalers'] = self.gnn_config.pna_scalers
            if self.gnn_config.pna_deg is not None:
                gnn_kwargs['deg'] = self.gnn_config.pna_deg

        # Transformer parameters
        elif self.gnn_config.layer_name == "TransformerConv":
            gnn_kwargs['heads'] = self.gnn_config.transformer_heads
            gnn_kwargs['concat'] = False

        # SAGE parameters
        elif self.gnn_config.layer_name == "SAGE":
            gnn_kwargs['aggr'] = self.gnn_config.sage_aggr

        # ChebConv parameters
        elif self.gnn_config.layer_name == "ChebConv":
            gnn_kwargs['K'] = self.gnn_config.cheb_k

        # ARMAConv parameters
        elif self.gnn_config.layer_name == "ARMAConv":
            gnn_kwargs['num_stacks'] = self.gnn_config.arma_num_stacks
            gnn_kwargs['num_layers'] = self.gnn_config.arma_num_layers

        # SGConv parameters
        elif self.gnn_config.layer_name == "SGConv":
            gnn_kwargs['K'] = self.gnn_config.sgconv_k

        # TAGConv parameters
        elif self.gnn_config.layer_name == "TAGConv":
            gnn_kwargs['K'] = self.gnn_config.tagconv_k

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key.startswith('gnn_'):
                gnn_kwargs[key[4:]] = value

        return gnn_kwargs

    def _extract_regressor_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Regressor-specific keyword arguments."""
        regressor_kwargs = {}

        for key, value in kwargs.items():
            if key.startswith('regressor_'):
                regressor_kwargs[key[10:]] = value

        return regressor_kwargs

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass through the complete pipeline.

        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features] (optional)
            batch: Batch assignment [num_nodes] (optional, defaults to single graph)
            return_embeddings: Whether to return intermediate embeddings

        Returns:
            predictions: Final predictions [batch_size]
            embeddings: Dictionary with node and graph embeddings (if return_embeddings=True)
        """
        # Handle single graph case
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Apply gradient clipping during training
        if self.training and self.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.gradient_clipping)

        # Get node embeddings from GNN
        node_embeddings = self.gnn(x, edge_index, edge_attr)

        # Get graph embeddings from pooling
        graph_embeddings = self.pooling(node_embeddings, batch)

        # Apply additional processing (skip batch norm if batch size is 1)
        if self.training and graph_embeddings.size(0) == 1 and isinstance(self.graph_batch_norm, nn.BatchNorm1d):
            # Skip batch norm for single sample during training
            pass
        else:
            graph_embeddings = self.graph_batch_norm(graph_embeddings)
        graph_embeddings = self.graph_dropout(graph_embeddings)

        # Get final predictions from regressor
        predictions = self.regressor(graph_embeddings)

        # Denormalize predictions if target statistics are available
        if self.target_mean is not None and self.target_std is not None:
            predictions = predictions * self.target_std + self.target_mean

        if return_embeddings:
            embeddings = {
                'node_embeddings': node_embeddings,
                'graph_embeddings': graph_embeddings
            }
            return predictions, embeddings

        return predictions

    def get_node_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract node-level embeddings from the GNN.

        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features] (optional)

        Returns:
            node_embeddings: Node embeddings [num_nodes, gnn_output_dim]
        """
        with torch.no_grad():
            node_embeddings = self.gnn(x, edge_index, edge_attr)
        return node_embeddings

    def get_graph_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract graph-level embeddings after pooling.

        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features] (optional)
            batch: Batch assignment [num_nodes] (optional)

        Returns:
            graph_embeddings: Graph embeddings [batch_size, pooling_output_dim]
        """
        # Handle single graph case
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        with torch.no_grad():
            # Get node embeddings
            node_embeddings = self.gnn(x, edge_index, edge_attr)

            # Get graph embeddings
            graph_embeddings = self.pooling(node_embeddings, batch)

            # Apply additional processing (skip batch norm if batch size is 1)
            if self.training and graph_embeddings.size(0) == 1 and isinstance(self.graph_batch_norm, nn.BatchNorm1d):
                # Skip batch norm for single sample during training
                pass
            else:
                graph_embeddings = self.graph_batch_norm(graph_embeddings)

        return graph_embeddings

    def get_all_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract all intermediate embeddings from the pipeline.

        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features] (optional)
            batch: Batch assignment [num_nodes] (optional)

        Returns:
            embeddings: Dictionary containing all embeddings
        """
        # Handle single graph case
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        with torch.no_grad():
            # Get node embeddings
            node_embeddings = self.gnn(x, edge_index, edge_attr)

            # Get graph embeddings (before additional processing)
            graph_embeddings_raw = self.pooling(node_embeddings, batch)

            # Get graph embeddings (after additional processing)
            if self.training and graph_embeddings_raw.size(0) == 1 and isinstance(self.graph_batch_norm, nn.BatchNorm1d):
                # Skip batch norm for single sample during training
                graph_embeddings_processed = graph_embeddings_raw
            else:
                graph_embeddings_processed = self.graph_batch_norm(
                    graph_embeddings_raw)

            return {
                'node_embeddings': node_embeddings,
                'graph_embeddings_raw': graph_embeddings_raw,
                'graph_embeddings_processed': graph_embeddings_processed
            }

    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Make predictions in evaluation mode.

        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features] (optional)
            batch: Batch assignment [num_nodes] (optional)

        Returns:
            predictions: Final predictions [batch_size]
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x, edge_index, edge_attr, batch)

    def get_num_parameters(self) -> Dict[str, int]:
        """Get the number of parameters in each component."""
        gnn_params = sum(p.numel() for p in self.gnn.parameters())
        pooling_params = sum(p.numel() for p in self.pooling.parameters())
        regressor_params = sum(p.numel() for p in self.regressor.parameters())

        return {
            'gnn': gnn_params,
            'pooling': pooling_params,
            'regressor': regressor_params,
            'other': sum(p.numel() for p in self.parameters()) - (gnn_params + pooling_params + regressor_params),
            'total': sum(p.numel() for p in self.parameters())
        }

    def get_config(self) -> Dict[str, Any]:
        """Get the complete configuration of the pipeline."""
        return {
            'node_features': self.node_features,
            'edge_features': self.edge_features,
            'gnn_config': self.gnn_config.__dict__,
            'pooling_config': self.pooling_config.__dict__,
            'regressor_config': self.regressor_config.__dict__,
            'global_dropout': self.global_dropout,
            'use_batch_norm': self.use_batch_norm,
            'target_mean': self.target_mean.item() if self.target_mean is not None else None,
            'target_std': self.target_std.item() if self.target_std is not None else None,
            'gradient_clipping': self.gradient_clipping
        }

    def set_target_normalization(self, target_mean: float, target_std: float):
        """Set target normalization parameters."""
        self.register_buffer('target_mean', torch.tensor(target_mean))
        self.register_buffer('target_std', torch.tensor(target_std))

    def save_config(self, filepath: str):
        """Save pipeline configuration to file."""
        import json
        config = self.get_config()
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'GDLPipeline':
        """Create pipeline from configuration dictionary."""
        return cls(**config)

    @classmethod
    def load_config(cls, filepath: str) -> 'GDLPipeline':
        """Load pipeline from configuration file."""
        import json
        with open(filepath, 'r') as f:
            config = json.load(f)
        return cls.from_config(config)


# Convenience functions for creating common pipeline configurations

def create_baseline_pipeline(
    node_features: int,
    edge_features: Optional[int] = None
) -> GDLPipeline:
    """Create a simple baseline pipeline for quick testing."""
    return GDLPipeline(
        node_features=node_features,
        edge_features=edge_features,
        gnn_config=GNNConfig(
            hidden_dim=64,
            num_layers=2,
            layer_name="GCN",
            dropout_rate=0.1
        ),
        pooling_config=PoolingConfig(pooling_type='mean'),
        regressor_config=RegressorConfig(
            regressor_type='linear',
            linear_dropout=0.1
        )
    )


def create_standard_pipeline(
    node_features: int,
    edge_features: Optional[int] = None
) -> GDLPipeline:
    """Create a standard pipeline recommended for most use cases."""
    return GDLPipeline(
        node_features=node_features,
        edge_features=edge_features,
        gnn_config=GNNConfig(
            hidden_dim=128,
            num_layers=4,
            layer_name="GINEConv",
            dropout_rate=0.15,
            use_residual=True,
            use_layer_norm=True
        ),
        pooling_config=PoolingConfig(pooling_type='attentional'),
        regressor_config=RegressorConfig(
            regressor_type='mlp',
            hidden_dims=[128, 64],
            mlp_dropout=0.15,
            normalization='batch'
        ),
        global_dropout=0.1,
        use_batch_norm=True
    )


def create_advanced_pipeline(
    node_features: int,
    edge_features: Optional[int] = None
) -> GDLPipeline:
    """Create an advanced pipeline for maximum performance."""
    return GDLPipeline(
        node_features=node_features,
        edge_features=edge_features,
        gnn_config=GNNConfig(
            hidden_dim=256,
            num_layers=5,
            layer_name="GINEConv",
            dropout_rate=0.15,
            linear_dropout_rate=0.4,
            use_residual=True,
            use_layer_norm=True
        ),
        pooling_config=PoolingConfig(pooling_type='attentional'),
        regressor_config=RegressorConfig(
            regressor_type='ensemble_mlp',
            hidden_dims=[256, 128, 64],
            mlp_dropout=0.1,
            ensemble_num_heads=3,
            ensemble_aggregation='weighted',
            normalization='layer'
        ),
        global_dropout=0.15,
        use_batch_norm=True,
        gradient_clipping=1.0
    )


def create_lightweight_pipeline(
    node_features: int,
    edge_features: Optional[int] = None
) -> GDLPipeline:
    """Create a lightweight pipeline for fast experimentation."""
    return GDLPipeline(
        node_features=node_features,
        edge_features=edge_features,
        gnn_config=GNNConfig(
            hidden_dim=64,
            num_layers=3,
            layer_name="SAGE",
            dropout_rate=0.1
        ),
        pooling_config=PoolingConfig(pooling_type='mean'),
        regressor_config=RegressorConfig(
            regressor_type='mlp',
            hidden_dims=[64, 32],
            mlp_dropout=0.1
        )
    )


def create_attention_pipeline(
    node_features: int,
    edge_features: Optional[int] = None
) -> GDLPipeline:
    """Create a pipeline focused on attention mechanisms."""
    return GDLPipeline(
        node_features=node_features,
        edge_features=edge_features,
        gnn_config=GNNConfig(
            hidden_dim=128,
            num_layers=4,
            layer_name="GAT",
            gat_heads=8,
            gat_concat=False,
            dropout_rate=0.15
        ),
        pooling_config=PoolingConfig(pooling_type='attentional'),
        regressor_config=RegressorConfig(
            regressor_type='attention_mlp',
            attention_hidden_dim=128,
            attention_num_heads=4,
            attention_dropout=0.15
        ),
        use_batch_norm=True
    )
