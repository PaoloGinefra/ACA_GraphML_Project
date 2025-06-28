"""
Multi-Objective Hyperparameter Optimization for GNN Pipeline using Optuna and Weights & Biases.

This script implements a comprehensive hyperparameter optimization system that optimizes
multiple objectives simultaneously:
1. Validation Mean Absolute Error (primary - minimize)
2. Model Memory Usage (minimize) 
3. Training Throughput (maximize)
4. Inference Latency (minimize)
5. Training Time (minimize)

Key Features:
- Multi-objective optimization using Optuna
- Remote experiment tracking with Weights & Biases
- Memory profiling and performance monitoring
- Comprehensive hyperparameter space covering all pipeline components
- Kaggle-compatible execution environment
"""

import os
import sys
import time
import psutil
import gc
import warnings
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import pickle
import json

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import optuna
from optuna.samplers import TPESampler
import wandb

# Memory profiling
import tracemalloc
from torch.profiler import profile, record_function, ProfilerActivity

# Scientific computing
import numpy as np
import pandas as pd

# Graph ML libraries
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset

# Local imports - using your existing ACAgraphML package
from src.pipeline_components import (
    DataAugmenter, 
    DimentionalityReduction,  # Note: using your original spelling
    create_augmentation_transforms,
    load_zinc_dataset,
    create_pipeline_from_config,
    GDLPipelineLightningModule
)
from src.performance_monitor import PerformanceMonitor
from src.utils import setup_environment, log_system_info
from src.supabase_integration import create_supabase_storage, get_or_create_study


@dataclass
class OptimizationConfig:
    """Configuration for the optimization process."""
    # Study configuration
    study_name: str = "gnn_multiobjective_optimization"
    n_trials: int = 100
    n_jobs: int = 1  # Kaggle typically has single GPU
    
    # Training configuration
    max_epochs: int = 50
    early_stopping_patience: int = 10
    batch_size_options: List[int] = None
    
    # Data configuration
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Performance monitoring
    memory_limit_gb: float = 12.0  # Kaggle GPU memory limit
    time_limit_hours: float = 9.0  # Kaggle time limit
    
    # Weights & Biases
    wandb_project: str = "gnn-hyperopt"
    wandb_entity: Optional[str] = None
    
    def __post_init__(self):
        if self.batch_size_options is None:
            self.batch_size_options = [16, 32, 64, 128]


class MultiObjectiveOptimizer:
    """
    Multi-objective hyperparameter optimizer for GNN pipeline.
    
    This class orchestrates the entire optimization process, managing:
    - Hyperparameter space definition
    - Model training and evaluation  
    - Performance monitoring
    - Multi-objective optimization with Optuna
    - Weights & Biases logging
    - Persistent storage with Supabase (optional)
    """
    
    def __init__(self, config: OptimizationConfig, use_supabase: bool = False, supabase_config: dict = None):
        self.config = config
        self.use_supabase = use_supabase
        self.supabase_config = supabase_config
        
        # Load datasets once
        print("Loading ZINC dataset...")
        self.train_dataset, self.val_dataset = load_zinc_dataset()
        print(f"Dataset loaded: {len(self.train_dataset)} train, {len(self.val_dataset)} val")
        
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor(
            memory_limit_gb=config.memory_limit_gb,
            time_limit_hours=config.time_limit_hours
        )
        
        # Initialize Optuna study
        if use_supabase and supabase_config:
            print("Setting up Supabase storage...")
            try:
                self.storage = create_supabase_storage(
                    supabase_url=supabase_config['url'],
                    supabase_password=supabase_config['password']
                )
                self.study = get_or_create_study(
                    storage=self.storage,
                    study_name=config.study_name,
                    directions=[
                        "minimize",  # Validation MAE
                        "minimize",  # Memory usage
                        "maximize",  # Throughput 
                        "minimize",  # Latency
                        "minimize"   # Training time
                    ]
                )
                print(f"✅ Connected to Supabase! Study has {len(self.study.trials)} existing trials.")
            except Exception as e:
                print(f"⚠️ Supabase connection failed: {e}")
                print("Falling back to in-memory storage...")
                self.study = optuna.create_study(
                    study_name=config.study_name,
                    directions=["minimize", "minimize", "maximize", "minimize", "minimize"],
                    sampler=TPESampler(seed=42)
                )
        else:
            # Use in-memory storage
            self.study = optuna.create_study(
                study_name=config.study_name,
                directions=["minimize", "minimize", "maximize", "minimize", "minimize"],
                sampler=TPESampler(seed=42)
            )
        
    def define_hyperparameter_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define the comprehensive hyperparameter space for all pipeline components.
        
        This function defines hyperparameters for:
        1. Data Augmentation
        2. Dimensionality Reduction
        3. GNN Architecture
        4. Training Configuration
        """
        
        # Data Augmentation hyperparameters
        augmentation_params = {
            'use_augmentation': trial.suggest_categorical('use_augmentation', [True, False]),
            'add_self_loops': trial.suggest_categorical('add_self_loops', [True, False]),
            'normalize_features': trial.suggest_categorical('normalize_features', [True, False]),
            'add_random_walk_pe': trial.suggest_categorical('add_random_walk_pe', [True, False]),
        }
        
        if augmentation_params['add_random_walk_pe']:
            augmentation_params['walk_length'] = trial.suggest_int('walk_length', 5, 20)
        
        # Dimensionality Reduction hyperparameters
        dim_reduction_params = {
            'use_dim_reduction': trial.suggest_categorical('use_dim_reduction', [True, False]),
            'explained_variance_ratio': trial.suggest_float('explained_variance_ratio', 0.85, 0.99),
        }
        
        # GNN Architecture hyperparameters
        gnn_params = {
            'gnn_type': trial.suggest_categorical('gnn_type', ['GCN', 'GIN', 'GINE', 'GraphSAGE', 'GAT']),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256, 512]),
            'num_layers': trial.suggest_int('num_layers', 2, 6),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'use_residual': trial.suggest_categorical('use_residual', [True, False]),
            'use_layer_norm': trial.suggest_categorical('use_layer_norm', [True, False]),
        }
        
        # Pooling hyperparameters
        pooling_params = {
            'pooling_type': trial.suggest_categorical('pooling_type', ['mean', 'max', 'add', 'attentional', 'set2set']),
        }
        
        if pooling_params['pooling_type'] in ['attentional', 'set2set']:
            pooling_params['processing_steps'] = trial.suggest_int('processing_steps', 1, 5)
        
        # Regressor hyperparameters
        regressor_params = {
            'regressor_type': trial.suggest_categorical('regressor_type', ['linear', 'mlp', 'residual_mlp', 'attention_mlp', 'ensemble_mlp']),
            'regressor_hidden_dims': [trial.suggest_categorical('regressor_hidden_1', [32, 64, 128, 256])],
            'regressor_activation': trial.suggest_categorical('regressor_activation', ['relu', 'gelu', 'swish']),
            'regressor_dropout': trial.suggest_float('regressor_dropout', 0.0, 0.5),
        }
        
        if regressor_params['regressor_type'] in ['mlp', 'residual_mlp', 'attention_mlp', 'ensemble_mlp']:
            regressor_params['regressor_hidden_dims'].append(
                trial.suggest_categorical('regressor_hidden_2', [16, 32, 64, 128])
            )
        
        # Training hyperparameters
        training_params = {
            'batch_size': trial.suggest_categorical('batch_size', self.config.batch_size_options),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd']),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            'scheduler': trial.suggest_categorical('scheduler', ['cosine', 'plateau', 'step', 'none']),
        }
        
        if training_params['optimizer'] == 'sgd':
            training_params['momentum'] = trial.suggest_float('momentum', 0.8, 0.99)
        
        if training_params['scheduler'] in ['step']:
            training_params['scheduler_step_size'] = trial.suggest_int('scheduler_step_size', 5, 20)
            training_params['scheduler_gamma'] = trial.suggest_float('scheduler_gamma', 0.1, 0.9)
        
        # Combine all parameters
        hyperparams = {
            **augmentation_params,
            **dim_reduction_params,
            **gnn_params,
            **pooling_params,
            **regressor_params,
            **training_params,
        }
        
        return hyperparams
    
    def objective_function(self, trial: optuna.Trial) -> Tuple[float, float, float, float, float]:
        """
        Objective function that evaluates a single hyperparameter configuration.
        
        Returns:
            Tuple of (val_mae, memory_usage_gb, throughput_samples_per_sec, latency_ms, training_time_minutes)
        """
        
        # Start memory tracking
        tracemalloc.start()
        start_time = time.time()
        
        try:
            # Get hyperparameters for this trial
            hyperparams = self.define_hyperparameter_space(trial)
            
            # Initialize Weights & Biases run
            wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=f"trial_{trial.number}",
                config=hyperparams,
                reinit=True
            )
            
            # Create Lightning logger
            logger = WandbLogger(experiment=wandb_run)
            
            # Build pipeline with current hyperparameters
            pipeline = self._build_pipeline(hyperparams)
            
            # Apply data augmentation and dimensionality reduction
            train_dataset_processed = self._preprocess_dataset(self.train_dataset, hyperparams)
            val_dataset_processed = self._preprocess_dataset(self.val_dataset, hyperparams)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset_processed, 
                batch_size=hyperparams['batch_size'],
                shuffle=True,
                num_workers=0  # Kaggle environment limitation
            )
            val_loader = DataLoader(
                val_dataset_processed,
                batch_size=hyperparams['batch_size'],
                shuffle=False,
                num_workers=0
            )
            
            # Setup trainer with callbacks - configure for both Kaggle and local
            checkpoint_dir = "/kaggle/working/checkpoints" if os.path.exists("/kaggle") else "./checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            callbacks = [
                EarlyStopping(
                    monitor='val_mae',
                    patience=self.config.early_stopping_patience,
                    mode='min'
                ),
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    monitor='val_mae',
                    mode='min',
                    save_top_k=1,
                    filename=f'trial_{trial.number}_best_{{val_mae:.4f}}',
                    save_last=True,
                    auto_insert_metric_name=False
                )
            ]
            
            trainer = pl.Trainer(
                max_epochs=self.config.max_epochs,
                callbacks=callbacks,
                logger=logger,
                enable_progress_bar=False,  # Reduce output in Kaggle
                enable_model_summary=False,
                deterministic=True,
                devices=1,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                precision='16-mixed' if torch.cuda.is_available() else 32,  # Use mixed precision on GPU
                gradient_clip_val=1.0,  # Gradient clipping for stability
            )
            
            # Train the model
            train_start = time.time()
            trainer.fit(pipeline, train_loader, val_loader)
            training_time = (time.time() - train_start) / 60  # in minutes
            
            # Get validation metrics
            val_results = trainer.validate(pipeline, val_loader, verbose=False)
            val_mae = val_results[0]['val_mae']
            
            # Measure inference performance
            throughput, latency = self._measure_inference_performance(pipeline, val_loader)
            
            # Measure memory usage
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            memory_usage_gb = peak_memory / (1024 ** 3)  # Convert to GB
            
            # Add GPU memory if available
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                memory_usage_gb += gpu_memory_gb
            
            # Log comprehensive metrics to wandb
            wandb.log({
                'trial_number': trial.number,
                'val_mae': val_mae,
                'memory_usage_gb': memory_usage_gb,
                'throughput_samples_per_sec': throughput,
                'latency_ms': latency,
                'training_time_minutes': training_time,
                'gpu_available': torch.cuda.is_available(),
            })
            
            # Clean up
            wandb.finish()
            del pipeline, train_loader, val_loader
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
            tracemalloc.stop()
            
            return val_mae, memory_usage_gb, throughput, latency, training_time
            
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {str(e)}")
            tracemalloc.stop()
            
            # Return worst possible values for failed trials
            return float('inf'), float('inf'), 0.0, float('inf'), float('inf')
    
    def _build_pipeline(self, hyperparams: Dict[str, Any]) -> pl.LightningModule:
        """Build the complete pipeline with given hyperparameters using your existing components."""
        
        # Create configuration dictionary for your existing GDLPipelineLightningModule
        pipeline_config = {
            'gnn_config': {
                'gnn_type': hyperparams['gnn_type'],
                'hidden_dim': hyperparams['hidden_dim'],
                'num_layers': hyperparams['num_layers'],
                'dropout': hyperparams['dropout'],
                'use_residual': hyperparams['use_residual'],
                'use_layer_norm': hyperparams['use_layer_norm'],
            },
            'pooling_config': {
                'pooling_type': hyperparams['pooling_type'],
                'processing_steps': hyperparams.get('processing_steps', 3),
            },
            'regressor_config': {
                'regressor_type': hyperparams['regressor_type'],
                'hidden_dims': hyperparams['regressor_hidden_dims'],
                'activation': hyperparams['regressor_activation'],
                'dropout': hyperparams['regressor_dropout'],
            },
            'training_config': {
                'learning_rate': hyperparams['learning_rate'],
                'optimizer': hyperparams['optimizer'],
                'weight_decay': hyperparams['weight_decay'],
                'scheduler': hyperparams['scheduler'],
                'momentum': hyperparams.get('momentum'),
                'scheduler_step_size': hyperparams.get('scheduler_step_size'),
                'scheduler_gamma': hyperparams.get('scheduler_gamma'),
            }
        }
        
        # Use your existing pipeline creation function
        pipeline = create_pipeline_from_config(pipeline_config)
        return pipeline
    
    def _preprocess_dataset(self, dataset: Dataset, hyperparams: Dict[str, Any]) -> Dataset:
        """Apply data augmentation and dimensionality reduction."""
        processed_dataset = dataset
        
        # Apply data augmentation if enabled
        if hyperparams['use_augmentation']:
            transforms = create_augmentation_transforms(hyperparams)
            augmenter = DataAugmenter(processed_dataset, transforms)
            processed_dataset = augmenter.augment()
        
        # Apply dimensionality reduction if enabled
        if hyperparams['use_dim_reduction']:
            dim_reducer = DimentionalityReduction(
                explained_variance_ratio=hyperparams['explained_variance_ratio']
            )
            processed_dataset = dim_reducer(processed_dataset)
        
        return processed_dataset
    
    def _measure_inference_performance(self, model: pl.LightningModule, dataloader: DataLoader) -> Tuple[float, float]:
        """Measure inference throughput and latency."""
        model.eval()
        model = model.cuda() if torch.cuda.is_available() else model
        
        # Warmup
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= 3:  # Just a few warmup iterations
                    break
                batch = batch.cuda() if torch.cuda.is_available() else batch
                _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        
        # Measure actual performance
        start_time = time.time()
        total_samples = 0
        latencies = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.cuda() if torch.cuda.is_available() else batch
                
                # Measure latency for this batch
                batch_start = time.time()
                _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                batch_time = time.time() - batch_start
                
                batch_size = batch.num_graphs
                total_samples += batch_size
                latencies.append(batch_time * 1000 / batch_size)  # ms per sample
        
        total_time = time.time() - start_time
        throughput = total_samples / total_time  # samples per second
        avg_latency = np.mean(latencies)  # ms per sample
        
        return throughput, avg_latency
    
    def run_optimization(self):
        """Run the complete multi-objective optimization."""
        print("Starting Multi-Objective Hyperparameter Optimization")
        print("=" * 60)
        
        # Log system information
        log_system_info()
        
        # Run optimization
        try:
            self.study.optimize(
                self.objective_function,
                n_trials=self.config.n_trials,
                n_jobs=self.config.n_jobs,
                timeout=self.config.time_limit_hours * 3600
            )
        except KeyboardInterrupt:
            print("Optimization interrupted by user")
        
        # Analyze and save results
        self._analyze_results()
    
    def _analyze_results(self):
        """Analyze optimization results and save summary."""
        print("\nOptimization completed!")
        print(f"Number of completed trials: {len(self.study.trials)}")
        
        # Get Pareto front
        pareto_trials = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                pareto_trials.append(trial)
        
        if not pareto_trials:
            print("No completed trials found!")
            return
        
        # Save detailed results
        results_df = pd.DataFrame([
            {
                'trial_number': trial.number,
                'val_mae': trial.values[0],
                'memory_gb': trial.values[1],
                'throughput_samples_sec': trial.values[2],
                'latency_ms': trial.values[3],
                'training_time_min': trial.values[4],
                **trial.params
            }
            for trial in pareto_trials
        ])
        
        results_df.to_csv('optimization_results.csv', index=False)
        
        # Find best trial for each objective
        best_mae_trial = min(pareto_trials, key=lambda t: t.values[0])
        best_memory_trial = min(pareto_trials, key=lambda t: t.values[1])
        best_throughput_trial = max(pareto_trials, key=lambda t: t.values[2])
        
        print(f"\nBest MAE: {best_mae_trial.values[0]:.4f} (Trial {best_mae_trial.number})")
        print(f"Best Memory: {best_memory_trial.values[1]:.2f} GB (Trial {best_memory_trial.number})")
        print(f"Best Throughput: {best_throughput_trial.values[2]:.2f} samples/sec (Trial {best_throughput_trial.number})")
        
        # Save study
        with open('optuna_study.pkl', 'wb') as f:
            pickle.dump(self.study, f)
        
        print(f"\nResults saved to optimization_results.csv and optuna_study.pkl")


def main():
    """Main execution function."""
    # Setup environment
    setup_environment()
    
    # Configuration
    config = OptimizationConfig(
        n_trials=50,  # Adjust based on available time
        max_epochs=30,
        early_stopping_patience=5,
        wandb_project="gnn-hyperopt-zinc",
    )
    
    # Check for Supabase configuration
    use_supabase = False
    supabase_config = None
    
    # Try to get Supabase credentials from environment variables (Kaggle Secrets)
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_password = os.getenv('SUPABASE_PASSWORD')
    
    if supabase_url and supabase_password:
        use_supabase = True
        supabase_config = {
            'url': supabase_url,
            'password': supabase_password
        }
        print("✅ Found Supabase credentials - will use persistent storage")
    else:
        print("⚠️ No Supabase credentials found - using in-memory storage")
        print("   To use Supabase, set SUPABASE_URL and SUPABASE_PASSWORD environment variables")
    
    # Initialize optimizer
    optimizer = MultiObjectiveOptimizer(config, use_supabase=use_supabase, supabase_config=supabase_config)
    
    # Run optimization
    optimizer.run_optimization()


if __name__ == "__main__":
    main()
