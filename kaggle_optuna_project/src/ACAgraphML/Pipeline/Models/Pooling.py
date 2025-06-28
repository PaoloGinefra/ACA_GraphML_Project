import torch
import torch.nn as nn
import torch_geometric.utils
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import Set2Set
from typing import Literal, Optional


class AttentionalPooling(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # Compute attention weights
        attention_weights = self.attention_layer(x)  # [num_nodes, 1]
        attention_weights = torch_geometric.utils.softmax(
            attention_weights.squeeze(), batch)  # [num_nodes]

        # Apply attention weights and sum
        # [num_nodes, hidden_dim]
        x_weighted = x * attention_weights.unsqueeze(-1)
        # [batch_size, hidden_dim]
        return torch_geometric.utils.scatter(x_weighted, batch, dim=0, reduce='sum')


class Pooling(nn.Module):
    def __init__(
        self,
        pooling_type: Literal['mean', 'max', 'add',
                              'attentional', 'set2set'] = 'mean',
        hidden_dim: int = 64,
        processing_steps: int = 3
    ) -> None:
        """
        Initialize the Pooling class with the desired pooling type.

        Args:
            pooling_type (str): The type of pooling to apply ('mean', 'max', 'add', 'attentional', 'set2set').
            hidden_dim (int): Hidden dimension size (used for attentional and set2set pooling).
            processing_steps (int): Number of processing steps for set2set pooling.
        """
        super().__init__()
        self.pooling_type: Literal['mean', 'max', 'add',
                                   'attentional', 'set2set'] = pooling_type

        if pooling_type == 'attentional':
            self.pooling_layer: Optional[nn.Module] = AttentionalPooling(
                hidden_dim=hidden_dim)
        elif pooling_type == 'set2set':
            self.pooling_layer = Set2Set(
                hidden_dim, processing_steps=processing_steps)
        else:
            self.pooling_layer = None

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pooling_type == 'mean':
            return global_mean_pool(x, batch)
        elif self.pooling_type == 'max':
            return global_max_pool(x, batch)
        elif self.pooling_type == 'add':
            return global_add_pool(x, batch)
        elif self.pooling_type == 'attentional':
            return self.pooling_layer(x, batch)  # type: ignore
        elif self.pooling_type == 'set2set':
            return self.pooling_layer(x, batch)  # type: ignore
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
