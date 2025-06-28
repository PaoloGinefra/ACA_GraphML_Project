"""
Pipeline components using the existing ACAgraphML package.

This module imports and configures your existing pipeline components.
"""

import sys
import os
from pathlib import Path

# Add the source directory to path to import ACAgraphML
# This works both locally and in Kaggle environment
current_dir = Path(__file__).parent
if (current_dir / "ACAgraphML").exists():
    # In Kaggle environment - ACAgraphML is in the same directory
    sys.path.insert(0, str(current_dir))
else:
    # In local environment - go up to find src
    project_root = current_dir.parent.parent
    src_path = project_root / "src"
    sys.path.insert(0, str(src_path))

# Import your existing components
from ACAgraphML.Pipeline.DataAugmenter import DataAugmenter
from ACAgraphML.Pipeline.DimentionalityReduction import DimentionalityReduction
from ACAgraphML.Pipeline.LightningModules.GDLPipelineLighningModule import GDLPipelineLightningModule
from ACAgraphML.Dataset import ZINC_Dataset
from ACAgraphML.Transforms import OneHotEncodeFeat

import torch_geometric.transforms as T
from torch_geometric.transforms import (
    AddSelfLoops, 
    NormalizeFeatures,
    AddRandomWalkPE,
    Compose
)
from torch_geometric.data import Dataset


def create_augmentation_transforms(hyperparams: dict) -> list:
    """
    Create list of augmentation transforms based on hyperparameters.
    """
    transforms = []
    
    if hyperparams.get('add_self_loops', False):
        transforms.append(AddSelfLoops())
    
    if hyperparams.get('normalize_features', False):
        transforms.append(NormalizeFeatures())
    
    if hyperparams.get('add_random_walk_pe', False):
        walk_length = hyperparams.get('walk_length', 10)
        transforms.append(AddRandomWalkPE(walk_length=walk_length))
    
    return transforms


def load_zinc_dataset():
    """
    Load ZINC dataset using your existing ZINC_Dataset class with train/val splits only.
    """
    # Use your existing ZINC dataset implementation
    NUM_NODE_FEATS = 28
    oneHotTransform = OneHotEncodeFeat(NUM_NODE_FEATS)
    
    # Load train and validation sets separately using your existing splits
    train_dataset = ZINC_Dataset.loadDatasetZINC(
        split='train',
        subset=True,  # Use subset for faster experimentation
        transform=oneHotTransform
    )
    
    val_dataset = ZINC_Dataset.loadDatasetZINC(
        split='val',
        subset=True,
        transform=oneHotTransform
    )
    
    return train_dataset, val_dataset


def _map_gnn_type_to_layer_name(gnn_type: str) -> str:
    """
    Map optimization GNN type names to your existing layer names.
    """
    mapping = {
        'GCN': 'GCN',
        'GIN': 'GINConv', 
        'GINE': 'GINEConv',
        'GraphSAGE': 'SAGE',
        'GAT': 'GAT'
    }
    return mapping.get(gnn_type, 'GCN')


def create_pipeline_from_config(config: dict) -> GDLPipelineLightningModule:
    """
    Create GDLPipelineLightningModule from configuration using your existing implementation.
    """
    # Extract configurations
    gnn_config = config.get('gnn_config', {})
    pooling_config = config.get('pooling_config', {})
    regressor_config = config.get('regressor_config', {})
    training_config = config.get('training_config', {})
    
    # ZINC dataset dimensions
    node_features = 28  # ZINC node features after one-hot encoding
    edge_features = 4   # ZINC edge features
    
    # Map hyperparameters to your existing module's expected format
    # Using "custom" pipeline configuration to have full control
    pipeline = GDLPipelineLightningModule(
        # Required parameters
        node_features=node_features,
        edge_features=edge_features,
        pipeline_config="custom",
        
        # Custom configurations
        gnn_config={
            'layer_name': _map_gnn_type_to_layer_name(gnn_config.get('gnn_type', 'GCN')),
            'hidden_dim': gnn_config.get('hidden_dim', 128),
            'num_layers': gnn_config.get('num_layers', 3),
            'dropout_rate': gnn_config.get('dropout', 0.1),
            'use_residual': gnn_config.get('use_residual', True),
            'use_layer_norm': gnn_config.get('use_layer_norm', True),
        },
        pooling_config={
            'pooling_type': pooling_config.get('pooling_type', 'mean'),
            'processing_steps': pooling_config.get('processing_steps', 3),
        },
        regressor_config={
            'regressor_type': regressor_config.get('regressor_type', 'mlp'),
            'hidden_dims': regressor_config.get('hidden_dims', [64]),
            'activation': regressor_config.get('activation', 'relu'),
            'mlp_dropout': regressor_config.get('dropout', 0.1),
        },
        
        # Training configuration
        optimizer=training_config.get('optimizer', 'adam'),
        lr=training_config.get('learning_rate', 1e-3),
        weight_decay=training_config.get('weight_decay', 1e-4),
        lr_scheduler=training_config.get('scheduler', 'none'),
        momentum=training_config.get('momentum', 0.9),
        
        # Additional scheduler parameters
        lr_scheduler_params={
            'step_size': training_config.get('scheduler_step_size', 10),
            'gamma': training_config.get('scheduler_gamma', 0.5),
        } if training_config.get('scheduler') == 'step' else {},
        
        # Loss and monitoring
        loss='mae',  # Using MAE for optimization
        monitor_metric='val_mae',
    )
    
    return pipeline
