"""
Utility functions for environment setup and logging.
"""

import os
import sys
import platform
import torch
import psutil
import warnings
from typing import Dict, Any
import random
import numpy as np


def setup_wandb_auth():
    """
    Setup Weights & Biases authentication for Kaggle environment.
    
    This function tries to authenticate with W&B using:
    1. WANDB_API_KEY environment variable (Kaggle Secrets)
    2. Existing wandb login
    3. Falls back to offline mode if needed
    """
    try:
        import wandb
        
        # Check if WANDB_API_KEY is available (from Kaggle Secrets)
        api_key = os.getenv('WANDB_API_KEY')
        
        if api_key:
            # Set the API key for wandb
            wandb.login(key=api_key)
            print("✅ W&B authentication successful with API key")
            return True
        else:
            # Try to use existing authentication
            try:
                wandb.login()
                print("✅ W&B authentication successful with existing credentials")
                return True
            except Exception as login_error:
                print(f"⚠️ W&B login failed: {login_error}")
                print("⚠️ Continuing in offline mode - logs will be saved locally")
                os.environ["WANDB_MODE"] = "offline"
                return False
                
    except ImportError:
        print("⚠️ wandb not installed")
        return False
    except Exception as e:
        print(f"⚠️ W&B setup failed: {e}")
        print("⚠️ Continuing in offline mode")
        os.environ["WANDB_MODE"] = "offline"
        return False


def setup_environment():
    """
    Setup the complete environment for reproducible experiments.
    """
    # Set random seeds for reproducibility
    set_random_seeds(42)
    
    # Setup W&B authentication
    setup_wandb_auth()
    
    # Configure warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    # Set environment variables for optimal performance
    os.environ['PYTHONHASHSEED'] = '42'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # PyTorch specific settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    print("Environment setup completed successfully!")


def set_random_seeds(seed: int = 42):
    """Set random seeds for all libraries to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For PyTorch Geometric
    try:
        import torch_geometric
        torch_geometric.seed_everything(seed)
    except ImportError:
        pass


def log_system_info():
    """Log comprehensive system information."""
    print("System Information")
    print("=" * 50)
    
    # Basic system info
    print(f"Platform: {platform.platform()}")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # CPU info
    print(f"CPU Count: {psutil.cpu_count()}")
    print(f"CPU Frequency: {psutil.cpu_freq().current if psutil.cpu_freq() else 'N/A'} MHz")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"Total Memory: {memory.total / (1024**3):.2f} GB")
    print(f"Available Memory: {memory.available / (1024**3):.2f} GB")
    
    # GPU info
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
    else:
        print("CUDA Available: No")
    
    # Library versions
    try:
        import torch_geometric
        print(f"PyTorch Geometric Version: {torch_geometric.__version__}")
    except ImportError:
        print("PyTorch Geometric: Not installed")
    
    try:
        import optuna
        print(f"Optuna Version: {optuna.__version__}")
    except ImportError:
        print("Optuna: Not installed")
    
    try:
        import wandb
        print(f"Weights & Biases Version: {wandb.__version__}")
    except ImportError:
        print("Weights & Biases: Not installed")
    
    print("=" * 50)


def check_kaggle_environment() -> bool:
    """Check if running in Kaggle environment."""
    return 'KAGGLE_WORKING_DIR' in os.environ


def get_kaggle_gpu_info() -> Dict[str, Any]:
    """Get GPU information specific to Kaggle environment."""
    info = {
        'kaggle_environment': check_kaggle_environment(),
        'gpu_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    return info


def optimize_for_kaggle():
    """Apply Kaggle-specific optimizations."""
    if not check_kaggle_environment():
        return
    
    # Kaggle-specific settings
    print("Applying Kaggle optimizations...")
    
    # Reduce memory usage
    torch.backends.cudnn.benchmark = True  # Enable for consistent input sizes
    
    # Set optimal number of workers (Kaggle has limited CPU cores)
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    
    print("Kaggle optimizations applied!")


def create_directory_structure(base_path: str):
    """Create necessary directory structure for the project."""
    directories = [
        'data',
        'models',
        'logs',
        'results',
        'checkpoints',
        'configs'
    ]
    
    for dir_name in directories:
        dir_path = os.path.join(base_path, dir_name)
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"Directory structure created at {base_path}")


def save_hyperparameters(hyperparams: Dict[str, Any], filepath: str):
    """Save hyperparameters to a JSON file."""
    import json
    
    # Convert any non-serializable objects to strings
    serializable_params = {}
    for key, value in hyperparams.items():
        if isinstance(value, (int, float, str, bool, list, dict)):
            serializable_params[key] = value
        else:
            serializable_params[key] = str(value)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_params, f, indent=2)
    
    print(f"Hyperparameters saved to {filepath}")


def load_hyperparameters(filepath: str) -> Dict[str, Any]:
    """Load hyperparameters from a JSON file."""
    import json
    
    with open(filepath, 'r') as f:
        hyperparams = json.load(f)
    
    return hyperparams


class ExperimentLogger:
    """Simple experiment logger for tracking key metrics."""
    
    def __init__(self, log_file: str = "experiment.log"):
        self.log_file = log_file
        
    def log(self, message: str):
        """Log a message with timestamp."""
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        
        print(log_entry.strip())
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics dictionary."""
        step_str = f" (Step {step})" if step is not None else ""
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.log(f"Metrics{step_str}: {metrics_str}")
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Log hyperparameters."""
        self.log("Hyperparameters:")
        for key, value in hyperparams.items():
            self.log(f"  {key}: {value}")


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_memory(bytes_value: float) -> str:
    """Format memory in bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"
