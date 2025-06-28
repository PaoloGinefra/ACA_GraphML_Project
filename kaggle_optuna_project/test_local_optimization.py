"""
Local testing script for the multi-objective optimization.

This script allows you to test the optimization locally before deploying to Kaggle.
It runs a small-scale version to verify everything works correctly.
"""

import os
import sys
import warnings
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import the optimization system
from optuna_multiobjective_optimization import MultiObjectiveOptimizer, OptimizationConfig
from src.utils import setup_environment


def test_local_optimization():
    """
    Run a small-scale local test of the optimization system.
    """
    print("🧪 Running Local Optimization Test")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Create test configuration (much smaller for local testing)
    config = OptimizationConfig(
        study_name="local_test_optimization",
        n_trials=3,  # Just 3 trials for testing
        max_epochs=5,  # Very short training
        early_stopping_patience=2,
        batch_size_options=[32, 64],  # Smaller batch sizes
        memory_limit_gb=8.0,  # Adjust based on your local GPU
        time_limit_hours=0.5,  # 30 minutes max
        wandb_project="gnn-hyperopt-local-test",
    )
    
    print(f"Configuration:")
    print(f"  - Trials: {config.n_trials}")
    print(f"  - Max epochs per trial: {config.max_epochs}")
    print(f"  - Time limit: {config.time_limit_hours} hours")
    print(f"  - Memory limit: {config.memory_limit_gb} GB")
    
    # Initialize optimizer
    try:
        print("\n📦 Initializing optimizer...")
        optimizer = MultiObjectiveOptimizer(config)
        print("✅ Optimizer initialized successfully!")
        
        # Test dataset loading
        print(f"📊 Dataset loaded:")
        print(f"   Train: {len(optimizer.train_dataset)}")
        print(f"   Val: {len(optimizer.val_dataset)}")
        
        # Test hyperparameter space
        print("\n🔧 Testing hyperparameter space...")
        import optuna
        
        # Create a sample trial with some parameters for testing
        sample_params = {
            'use_augmentation': False,
            'add_self_loops': True,
            'normalize_features': True,
            'add_random_walk_pe': False,
            'walk_length': 10,
            'use_dim_reduction': False,
            'explained_variance_ratio': 0.95,
            'gnn_type': 'GCN',
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.1,
            'use_residual': True,
            'use_layer_norm': True,
            'pooling_type': 'mean',
            'processing_steps': 3,
            'regressor_type': 'mlp',
            'regressor_hidden_1': 32,
            'regressor_hidden_2': 16,
            'regressor_activation': 'relu',
            'regressor_dropout': 0.1,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'optimizer': 'adam',
            'weight_decay': 1e-4,
            'scheduler': 'none',
            'momentum': 0.9,
            'scheduler_step_size': 10,
            'scheduler_gamma': 0.5
        }
        
        trial = optuna.trial.FixedTrial(sample_params)
        try:
            hyperparams = optimizer.define_hyperparameter_space(trial)
            print("✅ Hyperparameter space defined successfully!")
            print(f"   Sample hyperparameters: {len(hyperparams)} parameters")
        except Exception as e:
            print(f"❌ Error in hyperparameter space: {e}")
            return False
        
        # Test pipeline creation
        print("\n🏗️  Testing pipeline creation...")
        try:
            # Create a sample hyperparameter configuration
            sample_hyperparams = {
                'use_augmentation': False,
                'add_self_loops': True,
                'normalize_features': True,
                'add_random_walk_pe': False,
                'use_dim_reduction': False,
                'explained_variance_ratio': 0.95,
                'gnn_type': 'GCN',
                'hidden_dim': 64,
                'num_layers': 2,
                'dropout': 0.1,
                'use_residual': True,
                'use_layer_norm': True,
                'pooling_type': 'mean',
                'regressor_type': 'mlp',
                'regressor_hidden_dims': [32],
                'regressor_activation': 'relu',
                'regressor_dropout': 0.1,
                'batch_size': 32,
                'learning_rate': 1e-3,
                'optimizer': 'adam',
                'weight_decay': 1e-4,
                'scheduler': 'none'
            }
            
            pipeline = optimizer._build_pipeline(sample_hyperparams)
            print("✅ Pipeline created successfully!")
            print(f"   Model type: {type(pipeline).__name__}")
            
        except Exception as e:
            print(f"❌ Error creating pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Optionally run a single trial
        run_trial = input("\n🚀 Would you like to run a single optimization trial? (y/n): ").strip().lower()
        
        if run_trial == 'y':
            print("\n⏳ Running single optimization trial...")
            
            # Disable W&B for local testing if desired
            import wandb
            wandb.init(mode="disabled")  # Comment this line if you want W&B logging
            
            try:
                # Run just one trial
                study = optimizer.study
                study.optimize(optimizer.objective_function, n_trials=1)
                
                print("✅ Single trial completed successfully!")
                
                if len(study.trials) > 0:
                    trial = study.trials[0]
                    if trial.state == optuna.trial.TrialState.COMPLETE:
                        print(f"   Results: MAE={trial.values[0]:.4f}, Memory={trial.values[1]:.2f}GB")
                        print(f"   Throughput={trial.values[2]:.2f}, Latency={trial.values[3]:.2f}ms")
                        print(f"   Training time={trial.values[4]:.2f}min")
                    else:
                        print(f"   Trial failed with state: {trial.state}")
                
            except Exception as e:
                print(f"❌ Error during trial execution: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        print("\n🎉 Local test completed successfully!")
        print("\n📋 Next steps:")
        print("1. Configure Weights & Biases for remote logging")
        print("2. Set up Kaggle API credentials")
        print("3. Run: python deploy_to_kaggle.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during local test: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dependencies():
    """
    Check if all required dependencies are installed.
    """
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'torch',
        'torch_geometric', 
        'pytorch_lightning',
        'optuna',
        'wandb',
        'psutil',
        'numpy',
        'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies are installed!")
    return True


def main():
    """
    Main function for local testing.
    """
    print("🧪 Multi-Objective GNN Optimization - Local Test")
    print("=" * 60)
    
    # Check dependencies first
    if not check_dependencies():
        return
    
    # Run optimization test
    success = test_local_optimization()
    
    if success:
        print("\n🎯 Local testing completed successfully!")
        print("Your optimization system is ready for Kaggle deployment.")
    else:
        print("\n❌ Local testing failed.")
        print("Please fix the issues before deploying to Kaggle.")


if __name__ == "__main__":
    main()
