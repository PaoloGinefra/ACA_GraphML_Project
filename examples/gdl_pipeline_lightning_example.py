"""
Example usage of the GDLPipeline Lightning Module for hyperparameter optimization.

This example demonstrates how to use the GDLPipelineLightningModule for:
1. Basic training with different pipeline configurations
2. Hyperparameter optimization with Ray Tune or similar frameworks
3. Advanced optimization strategies
4. Model evaluation and analysis
"""

import os
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import optuna
from typing import Dict, Any

# Import your dataset and transforms
from ACAgraphML.Dataset import ZINC_Dataset
from ACAgraphML.Transforms import OneHotEncodeFeat
from ACAgraphML.Pipeline.LightningModules.GDLPipelineLightningModule import (
    GDLPipelineLightningModule,
    create_lightning_standard,
    create_lightning_advanced,
    create_lightning_custom
)
from ACAgraphML.Pipeline.Models.GDLPipeline import GNNConfig, PoolingConfig, RegressorConfig

# Constants
NUM_NODE_FEATS = 28
NUM_EDGE_FEATS = 4
BATCH_SIZE = 32
MAX_EPOCHS = 100


def prepare_data():
    """Prepare ZINC dataset."""
    # Custom transform
    oneHotTransform = OneHotEncodeFeat(NUM_NODE_FEATS)

    def ensure_float_transform(data):
        data = oneHotTransform(data)
        data.x = data.x.float()
        data.edge_attr = torch.nn.functional.one_hot(
            data.edge_attr.long(), num_classes=NUM_EDGE_FEATS
        ).float()
        return data

    # Load datasets
    train_dataset = ZINC_Dataset.SMALL_TRAIN.load(
        transform=ensure_float_transform)
    val_dataset = ZINC_Dataset.SMALL_VAL.load(transform=ensure_float_transform)
    test_dataset = ZINC_Dataset.SMALL_TEST.load(
        transform=ensure_float_transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def example_1_basic_usage():
    """Example 1: Basic usage with pre-defined configurations."""
    print("=== Example 1: Basic Usage ===")

    # Prepare data
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = prepare_data()

    # Calculate target statistics for normalization
    all_targets = torch.cat([train_dataset.y, val_dataset.y])
    target_mean = torch.mean(all_targets).item()
    target_std = torch.std(all_targets).item()

    # Create model with standard configuration
    model = create_lightning_standard(
        node_features=NUM_NODE_FEATS,
        edge_features=NUM_EDGE_FEATS,
        target_mean=target_mean,
        target_std=target_std,
        lr=1e-3,
        weight_decay=1e-4,
        lr_scheduler='cosine',
        lr_scheduler_params={'T_max': MAX_EPOCHS, 'eta_min': 1e-6}
    )

    # Setup callbacks
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

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=callbacks,
        accelerator='auto',
        devices=1,
        logger=TensorBoardLogger(
            'lightning_logs', name='gdl_pipeline_standard'),
        gradient_clip_val=1.0,
        gradient_clip_algorithm='norm',
        log_every_n_steps=10
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Test
    test_results = trainer.test(model, test_loader)
    print(f"Test MAE: {test_results[0]['test_mae']:.4f}")

    return model, trainer


def example_2_hyperparameter_optimization():
    """Example 2: Hyperparameter optimization with Optuna."""
    print("=== Example 2: Hyperparameter Optimization ===")

    def objective(trial):
        # Prepare data
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = prepare_data()

        # Calculate target statistics
        all_targets = torch.cat([train_dataset.y, val_dataset.y])
        target_mean = torch.mean(all_targets).item()
        target_std = torch.std(all_targets).item()

        # Suggest hyperparameters
        config = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
            'num_layers': trial.suggest_int('num_layers', 3, 6),
            'layer_name': trial.suggest_categorical('layer_name', ['GINEConv', 'GAT', 'GATv2', 'SAGE']),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.3),
            'pooling_type': trial.suggest_categorical('pooling_type', ['mean', 'attentional', 'set2set']),
            'regressor_type': trial.suggest_categorical('regressor_type', ['mlp', 'residual_mlp']),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'global_dropout': trial.suggest_float('global_dropout', 0.0, 0.2),
        }

        # Create custom configuration
        gnn_config = GNNConfig(
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            layer_name=config['layer_name'],
            dropout_rate=config['dropout_rate'],
            use_residual=True,
            use_layer_norm=True
        )

        pooling_config = PoolingConfig(
            pooling_type=config['pooling_type']
        )

        regressor_config = RegressorConfig(
            regressor_type=config['regressor_type'],
            hidden_dims=[config['hidden_dim'], config['hidden_dim'] // 2]
        )

        # Create model
        model = create_lightning_custom(
            node_features=NUM_NODE_FEATS,
            edge_features=NUM_EDGE_FEATS,
            gnn_config=gnn_config,
            pooling_config=pooling_config,
            regressor_config=regressor_config,
            global_dropout=config['global_dropout'],
            target_mean=target_mean,
            target_std=target_std,
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            lr_scheduler='cosine',
            lr_scheduler_params={'T_max': 50, 'eta_min': config['lr'] * 0.01}
        )

        # Setup trainer
        trainer = pl.Trainer(
            max_epochs=50,  # Shorter for optimization
            callbacks=[
                EarlyStopping(monitor='val_mae', patience=10,
                              mode='min', verbose=False)
            ],
            accelerator='auto',
            devices=1,
            logger=False,  # Disable logging for speed
            enable_progress_bar=False,
            gradient_clip_val=1.0
        )

        # Train
        trainer.fit(model, train_loader, val_loader)

        # Return validation MAE
        return trainer.callback_metrics['val_mae'].item()

    # Run optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)  # Adjust number of trials as needed

    print(f"Best MAE: {study.best_value:.4f}")
    print(f"Best hyperparameters: {study.best_params}")

    return study


def example_3_advanced_configuration():
    """Example 3: Advanced configuration with all features."""
    print("=== Example 3: Advanced Configuration ===")

    # Prepare data
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = prepare_data()

    # Calculate target statistics
    all_targets = torch.cat([train_dataset.y, val_dataset.y])
    target_mean = torch.mean(all_targets).item()
    target_std = torch.std(all_targets).item()

    # Create advanced model
    model = GDLPipelineLightningModule(
        node_features=NUM_NODE_FEATS,
        edge_features=NUM_EDGE_FEATS,
        pipeline_config="advanced",

        # Loss and optimization
        loss='mae',
        optimizer='adamw',
        lr=5e-4,
        weight_decay=1e-4,
        betas=(0.9, 0.999),

        # Learning rate scheduling
        lr_scheduler='cosine_restarts',
        lr_scheduler_params={'T_0': 10, 'T_mult': 2, 'eta_min': 1e-6},
        warmup_epochs=5,

        # Regularization
        gradient_clip_val=1.0,
        gradient_clip_algorithm='norm',

        # Target normalization
        target_mean=target_mean,
        target_std=target_std,

        # Monitoring
        monitor_metric='val_mae',
        log_embeddings=False,
        log_predictions=False,

        # Advanced features
        label_smoothing=0.0
    )

    # Setup comprehensive callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_mae',
            mode='min',
            save_top_k=5,
            filename='{epoch}-{val_mae:.4f}-{val_r2:.3f}',
            save_weights_only=True
        ),
        EarlyStopping(
            monitor='val_mae',
            patience=25,
            mode='min',
            verbose=True,
            min_delta=1e-4
        ),
        LearningRateMonitor(logging_interval='epoch'),
        pl.callbacks.ModelSummary(max_depth=2)
    ]

    # Setup trainer with advanced features
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=callbacks,
        accelerator='auto',
        devices=1,
        logger=[
            TensorBoardLogger('lightning_logs', name='gdl_pipeline_advanced'),
            # WandbLogger(project='gdl-pipeline', name='advanced-run')  # Uncomment if using wandb
        ],
        gradient_clip_val=1.0,
        gradient_clip_algorithm='norm',
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        precision=32,
        accumulate_grad_batches=1,
        enable_progress_bar=True
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Load best model and test
    best_model = GDLPipelineLightningModule.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )
    test_results = trainer.test(best_model, test_loader)

    # Print results
    print(f"Test Results:")
    print(f"  MAE: {test_results[0]['test_mae']:.4f}")
    print(f"  RMSE: {test_results[0]['test_rmse']:.4f}")
    print(f"  R²: {test_results[0]['test_r2']:.4f}")

    # Get model info
    config = best_model.get_pipeline_config()
    params = best_model.get_num_parameters()
    print(f"\nModel Info:")
    print(f"  Total parameters: {params['total']:,}")
    print(f"  GNN parameters: {params['gnn']:,}")
    print(f"  Pooling parameters: {params['pooling']:,}")
    print(f"  Regressor parameters: {params['regressor']:,}")

    return best_model, trainer


def example_4_embedding_extraction():
    """Example 4: Embedding extraction and analysis."""
    print("=== Example 4: Embedding Extraction ===")

    # Prepare data (small sample for demonstration)
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = prepare_data()

    # Create and train a simple model
    model = create_lightning_standard(
        node_features=NUM_NODE_FEATS,
        edge_features=NUM_EDGE_FEATS,
        lr=1e-3
    )

    # Quick training (few epochs for demo)
    trainer = pl.Trainer(
        max_epochs=5,
        logger=False,
        enable_progress_bar=True,
        accelerator='auto',
        devices=1
    )
    trainer.fit(model, train_loader, val_loader)

    # Extract embeddings from validation set
    model.eval()

    graph_embeddings = []
    targets = []

    with torch.no_grad():
        for batch in val_loader:
            # Get graph embeddings
            embeddings = model.get_embeddings(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                embedding_type='graph'
            )
            graph_embeddings.append(embeddings)
            targets.append(batch.y)

    # Concatenate all embeddings
    all_embeddings = torch.cat(graph_embeddings, dim=0)
    all_targets = torch.cat(targets, dim=0)

    print(f"Extracted {all_embeddings.shape[0]} graph embeddings")
    print(f"Embedding dimension: {all_embeddings.shape[1]}")

    # You can now use these embeddings for:
    # 1. Visualization (t-SNE, UMAP)
    # 2. Downstream ML tasks
    # 3. Similarity analysis
    # 4. Clustering

    return all_embeddings, all_targets


if __name__ == "__main__":
    # Set random seeds for reproducibility
    pl.seed_everything(42)

    # Run examples
    print("Running GDLPipeline Lightning Module Examples")
    print("=" * 50)

    # Example 1: Basic usage
    try:
        model1, trainer1 = example_1_basic_usage()
        print("✅ Example 1 completed successfully")
    except Exception as e:
        print(f"❌ Example 1 failed: {e}")

    print("\n" + "=" * 50)

    # Example 2: Hyperparameter optimization (comment out if optuna not available)
    try:
        study = example_2_hyperparameter_optimization()
        print("✅ Example 2 completed successfully")
    except Exception as e:
        print(
            f"❌ Example 2 failed (this is expected if optuna is not installed): {e}")

    print("\n" + "=" * 50)

    # Example 3: Advanced configuration
    try:
        model3, trainer3 = example_3_advanced_configuration()
        print("✅ Example 3 completed successfully")
    except Exception as e:
        print(f"❌ Example 3 failed: {e}")

    print("\n" + "=" * 50)

    # Example 4: Embedding extraction
    try:
        embeddings, targets = example_4_embedding_extraction()
        print("✅ Example 4 completed successfully")
    except Exception as e:
        print(f"❌ Example 4 failed: {e}")

    print("\n" + "=" * 50)
    print("All examples completed!")
