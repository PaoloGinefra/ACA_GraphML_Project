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
    """
    Configuration class for Graph Neural Network component.

    This dataclass encapsulates all parameters needed to configure the GNN layers
    in the pipeline, including layer-specific hyperparameters for different
    GNN architectures.

    Attributes:
        hidden_dim (int): Hidden dimension size for GNN layers. Default: 128
        output_dim (Optional[int]): Output dimension of GNN. If None, uses hidden_dim. Default: None
        num_layers (int): Number of GNN layers to stack. Default: 4
        layer_name (LayerName): Type of GNN layer to use (e.g., "GINEConv", "GAT"). Default: "GINEConv"
        dropout_rate (float): Dropout rate for GNN layers. Default: 0.1
        linear_dropout_rate (float): Dropout rate for linear layers in GNN. Default: 0.3
        use_residual (bool): Whether to use residual connections. Default: True
        use_layer_norm (bool): Whether to apply layer normalization. Default: True

        # Layer-specific parameters
        gat_heads (int): Number of attention heads for GAT layers. Default: 8
        gat_concat (bool): Whether to concatenate attention heads in GAT. Default: False
        pna_aggregators (List[str]): Aggregation functions for PNA layers. Default: ['mean', 'min', 'max', 'std']
        pna_scalers (List[str]): Scaling functions for PNA layers. Default: ['identity', 'amplification', 'attenuation']
        pna_deg (Optional[torch.Tensor]): Degree information for PNA layers. Default: None
        transformer_heads (int): Number of attention heads for Transformer layers. Default: 8
        sage_aggr (str): Aggregation method for SAGE layers ('mean', 'max', 'add'). Default: 'mean'
        cheb_k (int): Order of Chebyshev polynomials for ChebConv. Default: 3
        arma_num_stacks (int): Number of stacks for ARMA layers. Default: 1
        arma_num_layers (int): Number of layers per stack in ARMA. Default: 1
        sgconv_k (int): Number of hops for SGConv layers. Default: 2
        tagconv_k (int): Number of hops for TAGConv layers. Default: 3
    """
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
        """
        Post-initialization method to set default values for optional parameters.

        This method is automatically called after the dataclass is initialized
        and handles setting default values for parameters that depend on other
        configuration values or need complex initialization.
        """
        if self.output_dim is None:
            self.output_dim = self.hidden_dim
        if self.pna_aggregators is None:
            self.pna_aggregators = ['mean', 'min', 'max', 'std']
        if self.pna_scalers is None:
            self.pna_scalers = ['identity', 'amplification', 'attenuation']


@dataclass
class PoolingConfig:
    """
    Configuration class for graph pooling component.

    This dataclass defines parameters for converting node-level representations
    to graph-level representations through various pooling strategies.

    Attributes:
        pooling_type (Literal): Type of pooling operation to use.
            Options: 'mean', 'max', 'add', 'attentional', 'set2set'
            - 'mean': Simple mean pooling across all nodes in a graph
            - 'max': Max pooling across all nodes in a graph  
            - 'add': Sum pooling across all nodes in a graph
            - 'attentional': Learnable attention-based pooling
            - 'set2set': Set2Set pooling with LSTM processing
            Default: 'mean'
        processing_steps (int): Number of processing steps for set2set pooling.
            Only used when pooling_type='set2set'. Default: 3
        attention_hidden_multiplier (float): Multiplier for attention hidden dimension
            in attentional pooling. Hidden dim = input_dim * multiplier. Default: 1.0
    """
    pooling_type: Literal['mean', 'max', 'add',
                          'attentional', 'set2set'] = 'mean'
    processing_steps: int = 3  # For set2set pooling
    attention_hidden_multiplier: float = 1.0  # For attentional pooling


@dataclass
class RegressorConfig:
    """
    Configuration class for the regression head component.

    This dataclass defines parameters for the final prediction layers that
    convert graph embeddings to target predictions. Supports multiple
    regressor architectures from simple linear to complex ensemble models.

    Attributes:
        regressor_type (Literal): Type of regressor architecture.
            Options: 'linear', 'mlp', 'residual_mlp', 'attention_mlp', 'ensemble_mlp'
            - 'linear': Simple linear layer
            - 'mlp': Multi-layer perceptron
            - 'residual_mlp': MLP with residual connections
            - 'attention_mlp': MLP with self-attention mechanisms
            - 'ensemble_mlp': Ensemble of multiple MLP heads
            Default: 'mlp'

        # Linear regressor parameters
        linear_dropout (float): Dropout rate for linear regressor. Default: 0.0

        # MLP regressor parameters  
        hidden_dims (List[int]): Hidden layer dimensions for MLP. Default: [128, 64]
        mlp_dropout (float): Dropout rate for MLP layers. Default: 0.15
        normalization (Literal): Type of normalization ('none', 'batch', 'layer'). Default: 'batch'
        activation (str): Activation function name (e.g., 'relu', 'gelu'). Default: 'relu'

        # Residual MLP parameters
        residual_hidden_dim (int): Hidden dimension for residual MLP. Default: 128
        residual_num_layers (int): Number of layers in residual MLP. Default: 3
        residual_dropout (float): Dropout rate for residual MLP. Default: 0.1
        residual_normalization (Literal): Normalization for residual MLP ('batch', 'layer'). Default: 'layer'

        # Attention MLP parameters
        attention_hidden_dim (int): Hidden dimension for attention layers. Default: 128
        attention_num_heads (int): Number of attention heads. Default: 4
        attention_num_layers (int): Number of attention layers. Default: 2
        attention_dropout (float): Dropout rate for attention layers. Default: 0.1

        # Ensemble parameters
        ensemble_num_heads (int): Number of ensemble heads. Default: 3
        ensemble_aggregation (Literal): How to combine ensemble predictions ('mean', 'weighted'). Default: 'weighted'
    """
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
        """
        Post-initialization method to set default values for optional parameters.

        Sets default hidden dimensions for MLP if not provided.
        """
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

        # Additional processing layers after pooling
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

        # Freeze components if requested (useful for transfer learning)
        if freeze_gnn:
            for param in self.gnn.parameters():
                param.requires_grad = False
        if freeze_pooling:
            for param in self.pooling.parameters():
                param.requires_grad = False

        # Initialize weights using standard schemes
        self.apply(self._init_weights)

    def _extract_gnn_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract GNN-specific keyword arguments based on the selected layer type.

        This method processes the GNN configuration and extracts layer-specific
        parameters that need to be passed to the GNN model constructor.
        Different GNN layers require different parameters (e.g., GAT needs attention heads,
        PNA needs aggregators and scalers).

        Args:
            kwargs (Dict[str, Any]): Additional keyword arguments from pipeline initialization

        Returns:
            Dict[str, Any]: Dictionary of GNN-specific parameters to pass to GNNModel

        Note:
            Parameters are extracted based on the layer_name specified in gnn_config.
            Additional parameters can be passed with 'gnn_' prefix in kwargs.
        """
        gnn_kwargs = {}

        # GAT parameters - Graph Attention Networks
        if self.gnn_config.layer_name in ["GAT", "GATv2"]:
            gnn_kwargs['heads'] = self.gnn_config.gat_heads
            gnn_kwargs['concat'] = self.gnn_config.gat_concat

        # PNA parameters - Principal Neighbourhood Aggregation
        elif self.gnn_config.layer_name == "PNA":
            gnn_kwargs['aggregators'] = self.gnn_config.pna_aggregators
            gnn_kwargs['scalers'] = self.gnn_config.pna_scalers
            if self.gnn_config.pna_deg is not None:
                gnn_kwargs['deg'] = self.gnn_config.pna_deg

        # Transformer parameters - Graph Transformer
        elif self.gnn_config.layer_name == "TransformerConv":
            gnn_kwargs['heads'] = self.gnn_config.transformer_heads
            gnn_kwargs['concat'] = False

        # SAGE parameters - GraphSAGE
        elif self.gnn_config.layer_name == "SAGE":
            gnn_kwargs['aggr'] = self.gnn_config.sage_aggr

        # ChebConv parameters - Chebyshev Spectral Graph Convolution
        elif self.gnn_config.layer_name == "ChebConv":
            gnn_kwargs['K'] = self.gnn_config.cheb_k

        # ARMAConv parameters - ARMA Graph Convolution
        elif self.gnn_config.layer_name == "ARMAConv":
            gnn_kwargs['num_stacks'] = self.gnn_config.arma_num_stacks
            gnn_kwargs['num_layers'] = self.gnn_config.arma_num_layers

        # SGConv parameters - Simplified Graph Convolution
        elif self.gnn_config.layer_name == "SGConv":
            gnn_kwargs['K'] = self.gnn_config.sgconv_k

        # TAGConv parameters - Topology Adaptive Graph Convolution
        elif self.gnn_config.layer_name == "TAGConv":
            gnn_kwargs['K'] = self.gnn_config.tagconv_k

        # Add any additional kwargs with 'gnn_' prefix
        for key, value in kwargs.items():
            if key.startswith('gnn_'):
                gnn_kwargs[key[4:]] = value

        return gnn_kwargs

    def _extract_regressor_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract regressor-specific keyword arguments.

        This method filters additional keyword arguments that should be passed
        to the regressor component. Arguments with 'regressor_' prefix are
        extracted and passed to the Regressor constructor.

        Args:
            kwargs (Dict[str, Any]): Additional keyword arguments from pipeline initialization

        Returns:
            Dict[str, Any]: Dictionary of regressor-specific parameters
        """
        regressor_kwargs = {}

        for key, value in kwargs.items():
            if key.startswith('regressor_'):
                regressor_kwargs[key[10:]] = value

        return regressor_kwargs

    def _init_weights(self, module):
        """
        Initialize model weights using standard initialization schemes.

        This method applies Xavier/Glorot uniform initialization to linear layers
        and standard initialization to normalization layers. Called automatically
        during model construction via self.apply().

        Args:
            module: PyTorch module to initialize

        Note:
            - Linear layers: Xavier uniform initialization for weights, zeros for bias
            - BatchNorm/LayerNorm: Ones for weights, zeros for bias
        """
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
        # Handle single graph case - create batch tensor if not provided
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Apply gradient clipping during training if specified
        if self.training and self.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.gradient_clipping)

        # Step 1: Get node embeddings from GNN
        node_embeddings = self.gnn(x, edge_index, edge_attr)

        # Step 2: Get graph embeddings from pooling
        graph_embeddings = self.pooling(node_embeddings, batch)

        # Step 3: Apply additional processing (skip batch norm if batch size is 1)
        # This handles the case where batch normalization fails with single samples
        if self.training and graph_embeddings.size(0) == 1 and isinstance(self.graph_batch_norm, nn.BatchNorm1d):
            # Skip batch norm for single sample during training
            pass
        else:
            graph_embeddings = self.graph_batch_norm(graph_embeddings)
        graph_embeddings = self.graph_dropout(graph_embeddings)

        # Step 4: Get final predictions from regressor
        predictions = self.regressor(graph_embeddings)

        # Step 5: Denormalize predictions if target statistics are available
        if self.target_mean is not None and self.target_std is not None:
            predictions = predictions * self.target_std + self.target_mean

        # Return embeddings along with predictions if requested
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
        # Handle single graph case - create batch tensor if not provided
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        with torch.no_grad():
            # Get node embeddings from GNN
            node_embeddings = self.gnn(x, edge_index, edge_attr)

            # Get graph embeddings from pooling
            graph_embeddings = self.pooling(node_embeddings, batch)

            # Apply additional processing (skip batch norm if batch size is 1)
            # This handles the case where batch normalization fails with single samples
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
        # Handle single graph case - create batch tensor if not provided
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        with torch.no_grad():
            # Step 1: Get node embeddings from GNN
            node_embeddings = self.gnn(x, edge_index, edge_attr)

            # Step 2: Get graph embeddings (before additional processing)
            graph_embeddings_raw = self.pooling(node_embeddings, batch)

            # Step 3: Get graph embeddings (after additional processing)
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
        """
        Get the number of parameters in each component of the pipeline.

        This method provides a detailed breakdown of parameter counts for
        debugging, model analysis, and comparison purposes.

        Returns:
            Dict[str, int]: Dictionary with parameter counts for each component:
                - 'gnn': Parameters in the GNN component
                - 'pooling': Parameters in the pooling component  
                - 'regressor': Parameters in the regressor component
                - 'other': Parameters in other components (batch norm, etc.)
                - 'total': Total number of parameters
        """
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
        """
        Get the complete configuration of the pipeline.

        This method serializes all configuration parameters into a dictionary
        that can be used to recreate the exact same pipeline architecture.
        Useful for experiment tracking, model serialization, and reproducibility.

        Returns:
            Dict[str, Any]: Complete configuration dictionary including all
                component configurations and pipeline-level parameters
        """
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
        """
        Set target normalization parameters for prediction denormalization.

        This method allows setting or updating the normalization statistics
        that will be used to denormalize predictions back to the original scale.

        Args:
            target_mean (float): Mean of the target values in the training set
            target_std (float): Standard deviation of the target values in the training set

        Note:
            These parameters are registered as buffers, so they will be saved
            with the model state and moved to the appropriate device automatically.
        """
        self.register_buffer('target_mean', torch.tensor(target_mean))
        self.register_buffer('target_std', torch.tensor(target_std))

    def save_config(self, filepath: str):
        """
        Save pipeline configuration to a JSON file.

        Args:
            filepath (str): Path where to save the configuration file

        Note:
            The saved configuration can be loaded later using load_config()
            to recreate the exact same pipeline architecture.
        """
        import json
        config = self.get_config()
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'GDLPipeline':
        """
        Create pipeline from configuration dictionary.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing all
                parameters needed to initialize the pipeline

        Returns:
            GDLPipeline: New pipeline instance with the specified configuration
        """
        return cls(**config)

    @classmethod
    def load_config(cls, filepath: str) -> 'GDLPipeline':
        """
        Load pipeline from configuration file.

        Args:
            filepath (str): Path to the JSON configuration file

        Returns:
            GDLPipeline: New pipeline instance loaded from the configuration file
        """
        import json
        with open(filepath, 'r') as f:
            config = json.load(f)
        return cls.from_config(config)


# Convenience functions for creating common pipeline configurations

def create_baseline_pipeline(
    node_features: int,
    edge_features: Optional[int] = None
) -> GDLPipeline:
    """
    Create a simple baseline pipeline for quick testing and prototyping.

    This function creates a minimal pipeline with basic components suitable
    for initial experiments and baseline comparisons. Uses simple architectures
    with minimal regularization.

    Args:
        node_features (int): Number of input node features
        edge_features (Optional[int]): Number of edge features (None if not using edge features)

    Returns:
        GDLPipeline: Configured baseline pipeline

    Configuration:
        - GNN: 2-layer GCN with 64 hidden dimensions
        - Pooling: Simple mean pooling
        - Regressor: Linear regression with light dropout
    """
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
    """
    Create a standard pipeline recommended for most use cases.

    This function creates a well-balanced pipeline with proven architectures
    and hyperparameters that work well across a variety of molecular property
    prediction tasks. Good starting point for most applications.

    Args:
        node_features (int): Number of input node features
        edge_features (Optional[int]): Number of edge features (None if not using edge features)

    Returns:
        GDLPipeline: Configured standard pipeline

    Configuration:
        - GNN: 4-layer GINEConv with 128 hidden dimensions, residual connections, layer norm
        - Pooling: Learnable attentional pooling
        - Regressor: 2-layer MLP with batch normalization
        - Additional: Global dropout and batch normalization after pooling
    """
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
    """
    Create an advanced pipeline for maximum performance on challenging tasks.

    This function creates a sophisticated pipeline with advanced architectures
    and aggressive regularization. Designed for scenarios where maximum
    performance is needed and computational resources are available.

    Args:
        node_features (int): Number of input node features
        edge_features (Optional[int]): Number of edge features (None if not using edge features)

    Returns:
        GDLPipeline: Configured advanced pipeline

    Configuration:
        - GNN: 5-layer GINEConv with 256 hidden dimensions, heavy regularization
        - Pooling: Learnable attentional pooling
        - Regressor: Ensemble of 3 MLP heads with weighted aggregation
        - Additional: Gradient clipping, layer normalization, extensive dropout
    """
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
    """
    Create a lightweight pipeline for fast experimentation and resource-constrained environments.

    This function creates a computationally efficient pipeline with minimal
    parameters while maintaining reasonable performance. Ideal for rapid
    prototyping, debugging, or deployment in resource-limited settings.

    Args:
        node_features (int): Number of input node features
        edge_features (Optional[int]): Number of edge features (None if not using edge features)

    Returns:
        GDLPipeline: Configured lightweight pipeline

    Configuration:
        - GNN: 3-layer SAGE with 64 hidden dimensions
        - Pooling: Simple mean pooling
        - Regressor: Small 2-layer MLP
        - Minimal regularization for fast training
    """
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
    """
    Create a pipeline focused on attention mechanisms for interpretability and performance.

    This function creates a pipeline that emphasizes attention mechanisms
    throughout the architecture, providing both strong performance and
    interpretability through attention weights.

    Args:
        node_features (int): Number of input node features
        edge_features (Optional[int]): Number of edge features (None if not using edge features)

    Returns:
        GDLPipeline: Configured attention-focused pipeline

    Configuration:
        - GNN: 4-layer GAT with 8 attention heads
        - Pooling: Learnable attentional pooling
        - Regressor: Attention-based MLP with multi-head attention
        - Focus on interpretability through attention mechanisms
    """
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
