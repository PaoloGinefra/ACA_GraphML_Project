"""
Regressor Models for Molecular Property Prediction on ZINC Dataset

This module implements various regressor architectures for final molecular property prediction
after graph embeddings have been obtained from GNN + Pooling layers. Each regressor option
is optimized for different aspects of molecular regression tasks.

Available Regressor Types:

SIMPLE:
- Linear: Single linear layer, minimal parameters, good baseline
- LinearWithDropout: Linear + dropout for regularization

MODERATE COMPLEXITY:
- MLP: Multi-layer perceptron with ReLU activations
- MLPWithBatchNorm: MLP + batch normalization for stable training
- MLPWithLayerNorm: MLP + layer normalization, alternative to batch norm

ADVANCED:
- ResidualMLP: MLP with residual connections, helps with deeper networks
- AttentionMLP: Incorporates self-attention over hidden features
- EnsembleMLP: Multiple MLP heads with ensemble prediction

For ZINC dataset molecular property prediction:
- MLP/MLPWithBatchNorm: Recommended for best performance-complexity balance
- ResidualMLP: For deeper networks without vanishing gradients
- EnsembleMLP: When computational resources allow, highest accuracy
- Linear: For fast baseline comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, List, Optional


class LinearRegressor(nn.Module):
    """Simple linear regressor - good baseline."""

    def __init__(self, input_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.linear(x).squeeze(-1)


class MLPRegressor(nn.Module):
    """Multi-layer perceptron regressor with configurable depth and normalization."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.1,
        normalization: Literal['none', 'batch', 'layer'] = 'batch',
        activation: str = 'relu'
    ):
        super().__init__()

        self.normalization = normalization

        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims + [1]

        for i in range(len(dims) - 1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            # Skip activation and norm for final layer
            if i < len(dims) - 2:
                # Normalization
                if normalization == 'batch':
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                elif normalization == 'layer':
                    layers.append(nn.LayerNorm(dims[i + 1]))

                # Activation
                if activation == 'relu':
                    layers.append(nn.ReLU(inplace=True))
                elif activation == 'gelu':
                    layers.append(nn.GELU())
                elif activation == 'leaky_relu':
                    layers.append(nn.LeakyReLU(0.1, inplace=True))

                # Dropout
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)


class ResidualMLPRegressor(nn.Module):
    """MLP with residual connections for deeper networks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        normalization: Literal['batch', 'layer'] = 'layer'
    ):
        super().__init__()

        # Input projection to hidden dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = ResidualBlock(hidden_dim, dropout, normalization)
            self.residual_blocks.append(block)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)

        for block in self.residual_blocks:
            x = block(x)

        return self.output_proj(x).squeeze(-1)


class ResidualBlock(nn.Module):
    """Individual residual block for ResidualMLPRegressor."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        normalization: Literal['batch', 'layer'] = 'layer'
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            hidden_dim) if normalization == 'layer' else nn.BatchNorm1d(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.norm2 = nn.LayerNorm(
            hidden_dim * 2) if normalization == 'layer' else nn.BatchNorm1d(hidden_dim * 2)
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.norm1(x)
        x = F.relu(self.linear1(x))
        x = self.norm2(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return F.relu(x + residual)


class AttentionMLPRegressor(nn.Module):
    """MLP with self-attention over hidden features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # MLP layers after attention
        mlp_layers = []
        for i in range(num_layers):
            if i == 0:
                mlp_layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                ])
            else:
                mlp_layers.extend([
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.LayerNorm(hidden_dim // 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                ])
                hidden_dim = hidden_dim // 2

        mlp_layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)

        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]

        # Self-attention
        x, _ = self.attention(x, x, x)

        # Remove sequence dimension
        x = x.squeeze(1)  # [batch_size, hidden_dim]

        return self.mlp(x).squeeze(-1)


class EnsembleMLPRegressor(nn.Module):
    """Ensemble of multiple MLP regressors for improved accuracy."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 3,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.1,
        aggregation: Literal['mean', 'weighted'] = 'weighted'
    ):
        super().__init__()

        self.num_heads = num_heads
        self.aggregation = aggregation

        # Create multiple MLP heads
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            head = MLPRegressor(
                input_dim=input_dim,
                hidden_dims=hidden_dims.copy(),
                dropout=dropout,
                normalization='layer'
            )
            self.heads.append(head)

        # Weighted aggregation parameters
        if aggregation == 'weighted':
            self.head_weights = nn.Parameter(torch.ones(num_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get predictions from all heads
        predictions = []
        for head in self.heads:
            pred = head(x)
            predictions.append(pred)

        # [batch_size, num_heads]
        predictions = torch.stack(predictions, dim=1)

        if self.aggregation == 'mean':
            return predictions.mean(dim=1)
        elif self.aggregation == 'weighted':
            weights = F.softmax(self.head_weights, dim=0)
            return (predictions * weights.unsqueeze(0)).sum(dim=1)


class Regressor(nn.Module):
    """Main regressor class that supports multiple architectures."""

    def __init__(
        self,
        input_dim: int,
        regressor_type: Literal[
            'linear', 'mlp', 'residual_mlp', 'attention_mlp', 'ensemble_mlp'
        ] = 'mlp',
        # Linear options
        linear_dropout: float = 0.0,
        # MLP options
        hidden_dims: List[int] = [128, 64],
        mlp_dropout: float = 0.1,
        normalization: Literal['none', 'batch', 'layer'] = 'batch',
        activation: str = 'relu',
        # Residual MLP options
        residual_hidden_dim: int = 128,
        residual_num_layers: int = 3,
        residual_dropout: float = 0.1,
        # Attention MLP options
        attention_hidden_dim: int = 128,
        attention_num_heads: int = 4,
        attention_num_layers: int = 3,
        attention_dropout: float = 0.1,
        # Ensemble options
        ensemble_num_heads: int = 3,
        ensemble_aggregation: Literal['mean', 'weighted'] = 'weighted'
    ):
        """
        Initialize regressor with specified architecture.

        Args:
            input_dim: Dimension of input graph embeddings
            regressor_type: Type of regressor architecture

            # Architecture-specific parameters
            linear_dropout: Dropout for linear regressor
            hidden_dims: Hidden layer dimensions for MLP
            mlp_dropout: Dropout for MLP layers
            normalization: Type of normalization ('none', 'batch', 'layer')
            activation: Activation function ('relu', 'gelu', 'leaky_relu')
            residual_hidden_dim: Hidden dimension for residual MLP
            residual_num_layers: Number of residual blocks
            residual_dropout: Dropout for residual blocks
            attention_hidden_dim: Hidden dimension for attention MLP
            attention_num_heads: Number of attention heads
            attention_num_layers: Number of MLP layers after attention
            attention_dropout: Dropout for attention MLP
            ensemble_num_heads: Number of ensemble heads
            ensemble_aggregation: How to aggregate ensemble predictions
        """
        super().__init__()

        self.regressor_type = regressor_type
        self.input_dim = input_dim

        if regressor_type == 'linear':
            self.regressor = LinearRegressor(
                input_dim=input_dim,
                dropout=linear_dropout
            )

        elif regressor_type == 'mlp':
            self.regressor = MLPRegressor(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout=mlp_dropout,
                normalization=normalization,
                activation=activation
            )

        elif regressor_type == 'residual_mlp':
            self.regressor = ResidualMLPRegressor(
                input_dim=input_dim,
                hidden_dim=residual_hidden_dim,
                num_layers=residual_num_layers,
                dropout=residual_dropout,
                normalization='layer' if normalization == 'layer' else 'batch'
            )

        elif regressor_type == 'attention_mlp':
            self.regressor = AttentionMLPRegressor(
                input_dim=input_dim,
                hidden_dim=attention_hidden_dim,
                num_heads=attention_num_heads,
                num_layers=attention_num_layers,
                dropout=attention_dropout
            )

        elif regressor_type == 'ensemble_mlp':
            self.regressor = EnsembleMLPRegressor(
                input_dim=input_dim,
                num_heads=ensemble_num_heads,
                hidden_dims=hidden_dims,
                dropout=mlp_dropout,
                aggregation=ensemble_aggregation
            )

        else:
            raise ValueError(f"Unknown regressor type: {regressor_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through regressor.

        Args:
            x: Graph embeddings [batch_size, input_dim]

        Returns:
            predictions: Molecular property predictions [batch_size]
        """
        return self.regressor(x)


# Convenience functions for quick model creation
def create_baseline_regressor(input_dim: int) -> Regressor:
    """Create a simple baseline regressor for quick testing."""
    return Regressor(
        input_dim=input_dim,
        regressor_type='linear',
        linear_dropout=0.1
    )


def create_standard_regressor(input_dim: int) -> Regressor:
    """Create a standard MLP regressor - recommended for most use cases."""
    return Regressor(
        input_dim=input_dim,
        regressor_type='mlp',
        hidden_dims=[128, 64],
        mlp_dropout=0.15,
        normalization='batch',
        activation='relu'
    )


def create_advanced_regressor(input_dim: int) -> Regressor:
    """Create an advanced regressor for maximum performance."""
    return Regressor(
        input_dim=input_dim,
        regressor_type='ensemble_mlp',
        hidden_dims=[256, 128, 64],
        mlp_dropout=0.1,
        ensemble_num_heads=3,
        ensemble_aggregation='weighted'
    )
