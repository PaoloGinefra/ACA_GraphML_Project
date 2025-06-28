# GDLPipeline Lightning Module

A comprehensive PyTorch Lightning wrapper for the GDLPipeline, designed for extensive hyperparameter optimization and advanced training strategies.

## 🚀 Features

- **Complete GDLPipeline Integration**: Seamless wrapper around all GDLPipeline configurations
- **Advanced Optimization**: Multiple optimizers, learning rate schedulers, and regularization techniques
- **Comprehensive Logging**: Detailed metrics, embeddings, and prediction tracking
- **Hyperparameter Optimization Ready**: Built-in support for optimization frameworks like Optuna
- **Production Ready**: Robust error handling, target normalization, and model checkpointing
- **Flexible Configuration**: Support for all pipeline configurations plus custom setups

## 📦 Installation

Ensure you have the following dependencies:

```bash
# Core dependencies (required)
pip install torch pytorch-lightning torch-geometric

# Optional dependencies for advanced features
pip install optuna              # For hyperparameter optimization
pip install wandb              # For Weights & Biases logging
pip install tensorboard        # For TensorBoard logging
```

## 🎯 Quick Start

### Basic Usage

```python
from ACAgraphML.Pipeline.LightningModules.GDLPipelineLighningModule import create_lightning_standard
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

# Create model with standard configuration
model = create_lightning_standard(
    node_features=28,
    edge_features=4,
    lr=1e-3,
    weight_decay=1e-4,
    lr_scheduler='cosine'
)

# Setup trainer
trainer = pl.Trainer(
    max_epochs=100,
    accelerator='auto',
    devices=1,
    gradient_clip_val=1.0
)

# Train
trainer.fit(model, train_loader, val_loader)
```

### Custom Configuration

```python
from ACAgraphML.Pipeline.LightningModules.GDLPipelineLighningModule import GDLPipelineLightningModule
from ACAgraphML.Pipeline.Models.GDLPipeline import GNNConfig, PoolingConfig, RegressorConfig

# Create custom configuration
model = GDLPipelineLightningModule(
    node_features=28,
    edge_features=4,
    pipeline_config="custom",
    gnn_config=GNNConfig(
        hidden_dim=256,
        num_layers=5,
        layer_name="GINEConv",
        dropout_rate=0.15
    ),
    pooling_config=PoolingConfig(pooling_type='attentional'),
    regressor_config=RegressorConfig(regressor_type='ensemble_mlp'),
    lr=5e-4,
    lr_scheduler='cosine_restarts'
)
```

## ⚙️ Configuration Options

### Pipeline Configurations

- **`"baseline"`**: Simple GCN-based pipeline for quick testing
- **`"standard"`**: Recommended configuration for most use cases
- **`"advanced"`**: High-performance configuration with advanced features
- **`"lightweight"`**: Fast configuration for experimentation
- **`"attention"`**: Attention-focused architecture
- **`"custom"`**: Full customization using individual config objects

### Loss Functions

- **`"mae"`**: Mean Absolute Error (default)
- **`"mse"`**: Mean Squared Error
- **`"huber"`**: Huber Loss (robust to outliers)
- **`"smooth_l1"`**: Smooth L1 Loss

### Optimizers

- **`"adamw"`**: AdamW optimizer (default, recommended)
- **`"adam"`**: Adam optimizer
- **`"sgd"`**: SGD with momentum
- **`"rmsprop"`**: RMSprop optimizer

### Learning Rate Schedulers

- **`"cosine"`**: Cosine Annealing
- **`"plateau"`**: Reduce on Plateau
- **`"step"`**: Step LR
- **`"exponential"`**: Exponential LR
- **`"cosine_restarts"`**: Cosine Annealing with Warm Restarts
- **`"none"`**: No scheduler

## 🔍 Hyperparameter Optimization

The module is designed for extensive hyperparameter optimization:

### With Optuna

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    config = {
        'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
        'num_layers': trial.suggest_int('num_layers', 3, 6),
        'layer_name': trial.suggest_categorical('layer_name', ['GINEConv', 'GAT', 'SAGE']),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    }

    # Create and train model
    model = GDLPipelineLightningModule(
        node_features=28,
        edge_features=4,
        pipeline_config="custom",
        gnn_config=GNNConfig(
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            layer_name=config['layer_name']
        ),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    trainer = pl.Trainer(max_epochs=50, logger=False)
    trainer.fit(model, train_loader, val_loader)

    return trainer.callback_metrics['val_mae'].item()

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

### Key Hyperparameters for Optimization

```python
# GNN Architecture
'hidden_dim': [64, 128, 256, 512]
'num_layers': [3, 4, 5, 6]
'layer_name': ['GINEConv', 'GAT', 'GATv2', 'SAGE', 'PNA']
'dropout_rate': [0.1, 0.15, 0.2, 0.25, 0.3]

# Pooling Strategy
'pooling_type': ['mean', 'max', 'attentional', 'set2set']

# Regressor Configuration
'regressor_type': ['linear', 'mlp', 'residual_mlp', 'ensemble_mlp']
'hidden_dims': [[128, 64], [256, 128], [256, 128, 64]]

# Optimization
'lr': [1e-4, 5e-4, 1e-3, 5e-3]
'weight_decay': [0, 1e-6, 1e-5, 1e-4, 1e-3]
'optimizer': ['adam', 'adamw']

# Regularization
'global_dropout': [0.0, 0.1, 0.2]
'gradient_clip_val': [0.5, 1.0, 2.0]
```

## 📊 Advanced Features

### Target Normalization

```python
# Calculate target statistics
target_mean = torch.mean(train_targets).item()
target_std = torch.std(train_targets).item()

model = GDLPipelineLightningModule(
    node_features=28,
    edge_features=4,
    target_mean=target_mean,
    target_std=target_std
)
```

### Learning Rate Warmup

```python
model = GDLPipelineLightningModule(
    node_features=28,
    edge_features=4,
    warmup_epochs=10,
    lr_scheduler='cosine',
    lr_scheduler_params={'T_max': 100}
)
```

### Comprehensive Callbacks

```python
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

callbacks = [
    ModelCheckpoint(
        monitor='val_mae',
        mode='min',
        save_top_k=3,
        filename='{epoch}-{val_mae:.4f}',
        save_weights_only=True
    ),
    EarlyStopping(
        monitor='val_mae',
        patience=20,
        mode='min',
        verbose=True
    ),
    LearningRateMonitor(logging_interval='epoch')
]

trainer = pl.Trainer(
    max_epochs=100,
    callbacks=callbacks,
    gradient_clip_val=1.0
)
```

## 🔧 Embedding Extraction

```python
# After training, extract embeddings
model.eval()

# Get graph-level embeddings
graph_embeddings = model.get_embeddings(
    x, edge_index, edge_attr, batch,
    embedding_type='graph'
)

# Get node-level embeddings
node_embeddings = model.get_embeddings(
    x, edge_index, edge_attr,
    embedding_type='node'
)

# Get all embeddings
all_embeddings = model.get_embeddings(
    x, edge_index, edge_attr, batch,
    embedding_type='all'
)
```

## 📈 Monitoring and Logging

### Available Metrics

**Training Metrics:**

- `train_loss`: Training loss
- `train_mae`: Training MAE
- `train_rmse`: Training RMSE
- `train_r2`: Training R² score
- `lr`: Current learning rate

**Validation Metrics:**

- `val_loss`: Validation loss
- `val_mae`: Validation MAE
- `val_rmse`: Validation RMSE
- `val_r2`: Validation R² score
- `val_max_error`: Maximum absolute error
- `val_correlation`: Prediction-target correlation

**Test Metrics:**

- Same as validation metrics with `test_` prefix

### TensorBoard Logging

```python
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger('lightning_logs', name='gdl_pipeline')
trainer = pl.Trainer(logger=logger)
```

### Weights & Biases Logging

```python
from pytorch_lightning.loggers import WandbLogger

logger = WandbLogger(project='gdl-pipeline', name='experiment-1')
trainer = pl.Trainer(logger=logger)
```

## 🎛️ Model Analysis

### Parameter Counting

```python
# Get detailed parameter information
params = model.get_num_parameters()
print(f"Total parameters: {params['total']:,}")
print(f"GNN parameters: {params['gnn']:,}")
print(f"Pooling parameters: {params['pooling']:,}")
print(f"Regressor parameters: {params['regressor']:,}")
```

### Configuration Export

```python
# Get complete configuration
config = model.get_pipeline_config()

# Save for reproducibility
import json
with open('model_config.json', 'w') as f:
    json.dump(config, f, indent=2)
```

## 🔬 Best Practices

### For ZINC Dataset

1. **Start with Standard Configuration**:

   ```python
   model = create_lightning_standard(node_features=28, edge_features=4)
   ```

2. **Use Target Normalization**:

   ```python
   target_mean = torch.mean(train_targets).item()
   target_std = torch.std(train_targets).item()
   ```

3. **Enable Gradient Clipping**:

   ```python
   trainer = pl.Trainer(gradient_clip_val=1.0)
   ```

4. **Use Cosine Annealing**:

   ```python
   lr_scheduler='cosine'
   lr_scheduler_params={'T_max': max_epochs, 'eta_min': 1e-6}
   ```

5. **Monitor Multiple Metrics**:
   ```python
   callbacks = [
       ModelCheckpoint(monitor='val_mae', mode='min'),
       EarlyStopping(monitor='val_mae', patience=20)
   ]
   ```

### For Hyperparameter Optimization

1. **Start with a Small Grid**:

   - Focus on architecture (hidden_dim, num_layers, layer_name)
   - Then optimize learning parameters (lr, weight_decay)
   - Finally tune regularization (dropout, gradient clipping)

2. **Use Multi-Stage Optimization**:

   - Stage 1: Quick evaluation (25-50 epochs)
   - Stage 2: Detailed evaluation (100+ epochs) for best candidates

3. **Consider Ensemble Methods**:
   - Train multiple models with different architectures
   - Use `regressor_type='ensemble_mlp'` for internal ensembling

## 🐛 Troubleshooting

### Common Issues

1. **Memory Issues**:

   ```python
   # Reduce batch size or model size
   # Enable gradient accumulation
   trainer = pl.Trainer(accumulate_grad_batches=2)
   ```

2. **Slow Convergence**:

   ```python
   # Try different learning rates or schedulers
   # Use warmup for complex architectures
   warmup_epochs=10
   ```

3. **Overfitting**:

   ```python
   # Increase regularization
   global_dropout=0.2
   weight_decay=1e-3
   # Use early stopping
   EarlyStopping(patience=15)
   ```

4. **Underfitting**:
   ```python
   # Increase model capacity
   hidden_dim=256
   num_layers=5
   # Reduce regularization
   dropout_rate=0.1
   ```

## 📝 Examples

See `examples/gdl_pipeline_lightning_example.py` for comprehensive usage examples including:

1. Basic training with different configurations
2. Hyperparameter optimization with Optuna
3. Advanced feature usage
4. Embedding extraction and analysis

## 🤝 Contributing

When extending the Lightning module:

1. Maintain backward compatibility
2. Add comprehensive tests
3. Update documentation
4. Follow the existing code style
5. Test with ZINC dataset

## 📄 License

This module is part of the ACA GraphML project. See the main project LICENSE file for details.
