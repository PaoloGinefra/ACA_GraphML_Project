# Graph Deep Learning Pipeline (GDLPipeline)

A comprehensive, fully parametrized Graph Deep Learning Pipeline for molecular property prediction on the ZINC dataset. This pipeline combines Graph Neural Networks (GNNs), graph pooling mechanisms, and regression heads into a unified, highly configurable framework designed for extensive hyperparameter tuning.

## 🚀 Features

- **Fully Parametrized**: Every component is configurable for hyperparameter optimization
- **Multiple GNN Architectures**: Support for 13+ GNN layer types (GINEConv, GAT, PNA, etc.)
- **Various Pooling Strategies**: Mean, max, attention, Set2Set pooling options
- **Multiple Regressor Types**: Linear, MLP, Residual, Attention, and Ensemble regressors
- **Embedding Extraction**: Methods to extract node and graph-level embeddings
- **Configuration Management**: Save/load pipeline configurations
- **Production Ready**: Built-in normalization, regularization, and monitoring

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Configuration Options](#configuration-options)
- [Usage Examples](#usage-examples)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Embedding Extraction](#embedding-extraction)
- [API Reference](#api-reference)
- [Examples](#examples)

## 🛠️ Installation

```bash
# Navigate to your project directory
cd path/to/ACA_GraphML_Project

# The GDLPipeline is part of the ACAgraphML package
# Ensure you have the required dependencies installed
pip install torch torch-geometric torch-scatter torch-sparse
```

## ⚡ Quick Start

```python
from ACAgraphML.Pipeline.Models.GDLPipeline import create_standard_pipeline
import torch

# Create a standard pipeline for ZINC dataset
pipeline = create_standard_pipeline(node_features=28, edge_features=4)

# Prepare your data (example with dummy data)
x = torch.randn(20, 28)          # Node features
edge_index = torch.randint(0, 20, (2, 40))  # Edge connectivity
edge_attr = torch.randn(40, 4)   # Edge features
batch = torch.zeros(20, dtype=torch.long)   # Batch assignment

# Forward pass
predictions = pipeline(x, edge_index, edge_attr, batch)
print(f"Predictions: {predictions}")

# Extract embeddings
node_embeddings = pipeline.get_node_embeddings(x, edge_index, edge_attr)
graph_embeddings = pipeline.get_graph_embeddings(x, edge_index, edge_attr, batch)
```

## 🏗️ Architecture Overview

The GDLPipeline consists of three main components:

```
Input Graph → [GNN Layers] → [Pooling] → [Regressor] → Prediction
     ↓              ↓            ↓
Node Features → Node Embeddings → Graph Embeddings
```

### 1. GNN Component

Extracts node-level features using various graph neural network architectures:

**Available Architectures** (ordered by complexity):

- **Simple**: SGConv, GraphConv, GCN
- **Medium**: SAGE, GINConv, ChebConv, ARMAConv, TAGConv
- **Advanced**: GAT, GATv2, TransformerConv, GINEConv, PNA

### 2. Pooling Component

Converts node embeddings to graph-level representations:

- **Global Pooling**: mean, max, add
- **Learnable Pooling**: attentional, Set2Set

### 3. Regressor Component

Final prediction from graph embeddings:

- **Simple**: Linear
- **Standard**: MLP with various configurations
- **Advanced**: Residual MLP, Attention MLP, Ensemble MLP

## ⚙️ Configuration Options

### GNN Configuration

```python
from ACAgraphML.Pipeline.Models.GDLPipeline import GNNConfig

gnn_config = GNNConfig(
    hidden_dim=128,              # Hidden dimension size
    output_dim=128,              # Output dimension (None = hidden_dim)
    num_layers=4,                # Number of GNN layers
    layer_name="GINEConv",       # GNN architecture
    dropout_rate=0.15,           # Dropout rate
    use_residual=True,           # Use residual connections
    use_layer_norm=True,         # Use layer normalization
    # Architecture-specific parameters
    gat_heads=8,                 # GAT attention heads
    pna_aggregators=['mean', 'max', 'min', 'std']  # PNA aggregators
)
```

### Pooling Configuration

```python
from ACAgraphML.Pipeline.Models.GDLPipeline import PoolingConfig

pooling_config = PoolingConfig(
    pooling_type='attentional',  # Pooling strategy
    processing_steps=3           # Set2Set processing steps
)
```

### Regressor Configuration

```python
from ACAgraphML.Pipeline.Models.GDLPipeline import RegressorConfig

regressor_config = RegressorConfig(
    regressor_type='mlp',        # Regressor architecture
    hidden_dims=[128, 64],       # Hidden layer sizes
    mlp_dropout=0.15,           # Dropout rate
    normalization='batch',       # Normalization type
    activation='relu'            # Activation function
)
```

## 🎯 Usage Examples

### Basic Usage

```python
from ACAgraphML.Pipeline.Models.GDLPipeline import GDLPipeline, GNNConfig, PoolingConfig, RegressorConfig

# Create custom pipeline
pipeline = GDLPipeline(
    node_features=28,
    edge_features=4,
    gnn_config=GNNConfig(
        hidden_dim=256,
        num_layers=5,
        layer_name="GINEConv"
    ),
    pooling_config=PoolingConfig(pooling_type='attentional'),
    regressor_config=RegressorConfig(regressor_type='ensemble_mlp')
)
```

### Training Loop

```python
import torch.optim as optim
import torch.nn.functional as F

# Setup training
optimizer = optim.Adam(pipeline.parameters(), lr=0.001)
criterion = F.mse_loss

# Training step
pipeline.train()
predictions = pipeline(x, edge_index, edge_attr, batch)
loss = criterion(predictions, targets)
loss.backward()
optimizer.step()
```

### Convenience Functions

```python
from ACAgraphML.Pipeline.Models.GDLPipeline import (
    create_baseline_pipeline,    # Simple baseline
    create_standard_pipeline,    # Recommended default
    create_advanced_pipeline,    # High performance
    create_lightweight_pipeline, # Fast experiments
    create_attention_pipeline    # Attention-focused
)

# Quick pipeline creation
pipeline = create_advanced_pipeline(node_features=28, edge_features=4)
```

## 🔍 Hyperparameter Tuning

The GDLPipeline is designed for extensive hyperparameter optimization:

```python
# Example hyperparameter grid
hyperparameters = {
    'gnn_config': {
        'hidden_dim': [64, 128, 256],
        'num_layers': [3, 4, 5],
        'layer_name': ['GINEConv', 'GAT', 'PNA'],
        'dropout_rate': [0.1, 0.15, 0.2]
    },
    'pooling_config': {
        'pooling_type': ['mean', 'attentional', 'set2set']
    },
    'regressor_config': {
        'regressor_type': ['mlp', 'residual_mlp', 'ensemble_mlp'],
        'hidden_dims': [[128, 64], [256, 128, 64]]
    }
}

# Create pipeline from hyperparameters
for config in hyperparameter_combinations:
    pipeline = GDLPipeline(
        node_features=28,
        edge_features=4,
        **config
    )
    # Train and evaluate...
```

## 🧠 Embedding Extraction

Extract intermediate representations for analysis:

```python
# Extract node-level embeddings
node_embeddings = pipeline.get_node_embeddings(x, edge_index, edge_attr)
print(f"Node embeddings: {node_embeddings.shape}")  # [num_nodes, hidden_dim]

# Extract graph-level embeddings
graph_embeddings = pipeline.get_graph_embeddings(x, edge_index, edge_attr, batch)
print(f"Graph embeddings: {graph_embeddings.shape}")  # [batch_size, pooling_dim]

# Extract all intermediate embeddings
all_embeddings = pipeline.get_all_embeddings(x, edge_index, edge_attr, batch)
print("Available embeddings:", list(all_embeddings.keys()))
```

## 📊 Model Analysis

```python
# Get parameter counts
params = pipeline.get_num_parameters()
print(f"Total parameters: {params['total']:,}")
print(f"GNN: {params['gnn']:,}, Pooling: {params['pooling']:,}, Regressor: {params['regressor']:,}")

# Get configuration
config = pipeline.get_config()
print("Current configuration:", config)

# Save/load configuration
pipeline.save_config('my_config.json')
new_pipeline = GDLPipeline.load_config('my_config.json')
```

## 🎛️ Advanced Features

### Target Normalization

```python
# Set target normalization parameters
pipeline.set_target_normalization(target_mean=2.5, target_std=1.2)

# Predictions will be automatically denormalized
predictions = pipeline(x, edge_index, edge_attr, batch)
```

### Gradient Clipping

```python
pipeline = GDLPipeline(
    node_features=28,
    edge_features=4,
    gradient_clipping=1.0,  # Clip gradients to max norm of 1.0
    **configs
)
```

### Component Freezing

```python
pipeline = GDLPipeline(
    node_features=28,
    edge_features=4,
    freeze_gnn=True,      # Freeze GNN parameters
    freeze_pooling=False, # Keep pooling trainable
    **configs
)
```

## 📈 Performance Recommendations

### For ZINC Dataset:

1. **Best Performance**:

   - GNN: GINEConv or PNA
   - Pooling: Attentional
   - Regressor: Ensemble MLP

2. **Balanced Performance/Speed**:

   - GNN: GINEConv or GAT
   - Pooling: Attentional or Mean
   - Regressor: MLP with batch norm

3. **Fast Experimentation**:
   - GNN: SAGE or GCN
   - Pooling: Mean
   - Regressor: Linear or simple MLP

## 📁 File Structure

```
src/ACAgraphML/Pipeline/Models/
├── GDLPipeline.py          # Main pipeline implementation
├── GNNmodel.py             # GNN component
├── Pooling.py              # Pooling component
└── Regressor.py            # Regressor component

examples/
├── test_gdl_pipeline.py         # Comprehensive test suite
└── zinc_integration_example.py  # Real ZINC dataset example
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_gdl_pipeline.py
```

Test with real ZINC data:

```bash
python zinc_integration_example.py
```

## 🚨 Common Issues and Solutions

### 1. BatchNorm with Single Graphs

The pipeline automatically handles batch normalization issues with single graphs by skipping batch norm when batch size is 1 during training.

### 2. Memory Issues with Large Graphs

Use gradient checkpointing and smaller batch sizes:

```python
pipeline = GDLPipeline(
    # ... configs ...
    global_dropout=0.2,  # Higher dropout for regularization
    gradient_clipping=1.0  # Prevent gradient explosion
)
```

### 3. Edge Feature Compatibility

Ensure your GNN architecture supports edge features:

```python
# These layers support edge features:
edge_supported = ["GAT", "GATv2", "GINEConv", "PNA", "TransformerConv"]

# These layers don't use edge features:
no_edge_support = ["GCN", "SAGE", "GINConv", "ChebConv", "SGConv"]
```

## 📖 API Reference

### GDLPipeline Class

#### Constructor

```python
GDLPipeline(
    node_features: int,
    edge_features: Optional[int] = None,
    gnn_config: Optional[Union[GNNConfig, Dict]] = None,
    pooling_config: Optional[Union[PoolingConfig, Dict]] = None,
    regressor_config: Optional[Union[RegressorConfig, Dict]] = None,
    global_dropout: float = 0.0,
    use_batch_norm: bool = False,
    target_mean: Optional[float] = None,
    target_std: Optional[float] = None,
    gradient_clipping: Optional[float] = None,
    freeze_gnn: bool = False,
    freeze_pooling: bool = False
)
```

#### Methods

- `forward(x, edge_index, edge_attr, batch, return_embeddings=False)`: Main forward pass
- `get_node_embeddings(x, edge_index, edge_attr)`: Extract node embeddings
- `get_graph_embeddings(x, edge_index, edge_attr, batch)`: Extract graph embeddings
- `get_all_embeddings(x, edge_index, edge_attr, batch)`: Extract all embeddings
- `predict(x, edge_index, edge_attr, batch)`: Inference mode prediction
- `get_num_parameters()`: Get parameter counts
- `get_config()`: Get configuration dictionary
- `save_config(filepath)`: Save configuration to file
- `set_target_normalization(mean, std)`: Set target normalization

## 🤝 Contributing

When contributing to the GDLPipeline:

1. Ensure backward compatibility
2. Add comprehensive tests for new features
3. Update documentation
4. Follow the existing code style
5. Test with ZINC dataset when possible

## 📄 License

This project is part of the ACA GraphML project. See the main project LICENSE file for details.

## 🙏 Acknowledgments

- Built on PyTorch Geometric
- Inspired by molecular property prediction literature
- Designed for the ZINC dataset benchmark
- Optimized for hyperparameter tuning workflows

---

For more examples and advanced usage, see the `examples/` directory and the comprehensive test suite in `test_gdl_pipeline.py`.
