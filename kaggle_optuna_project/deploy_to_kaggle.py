"""
Kaggle deployment script for uploading and running the hyperparameter optimization.

This script handles:
1. Creating a Kaggle dataset with all necessary files
2. Creating and submitting a Kaggle notebook for execution
3. Monitoring the execution status
"""

import os
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import time

# Kaggle API
from kaggle.api.kaggle_api_extended import KaggleApi


class KaggleDeployer:
    """
    Handles deployment of the hyperparameter optimization to Kaggle.
    """
    
    def __init__(self, project_path: str, kaggle_username: str):
        self.project_path = Path(project_path)
        self.kaggle_username = kaggle_username
        self.api = KaggleApi()
        self.api.authenticate()
        
        # Dataset and kernel identifiers
        self.dataset_slug = f"{kaggle_username}/gnn-hyperopt-pipeline"
        self.kernel_slug = f"{kaggle_username}/gnn-multiobjective-optimization"
        
    def prepare_dataset(self) -> str:
        """
        Prepare and upload dataset containing all project files.
        """
        print("Preparing dataset for Kaggle upload...")
        
        # Create temporary directory for dataset
        dataset_dir = self.project_path / "kaggle_dataset"
        dataset_dir.mkdir(exist_ok=True)
        
        # Copy all necessary files
        files_to_copy = [
            "src/",
            "configs/",
            "requirements.txt",
            "optuna_multiobjective_optimization.py"
        ]
        
        for file_path in files_to_copy:
            src = self.project_path / file_path
            dst = dataset_dir / file_path
            
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        
        # Create dataset metadata
        dataset_metadata = {
            "title": "GNN Hyperparameter Optimization Pipeline",
            "id": self.dataset_slug,
            "licenses": [{"name": "MIT"}],
            "resources": []
        }
        
        # Add all files to resources
        for file_path in dataset_dir.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(dataset_dir)
                dataset_metadata["resources"].append({
                    "path": str(rel_path),
                    "description": f"Project file: {rel_path}"
                })
        
        # Save metadata
        with open(dataset_dir / "dataset-metadata.json", "w") as f:
            json.dump(dataset_metadata, f, indent=2)
        
        # Upload dataset
        try:
            self.api.dataset_create_new(
                folder=str(dataset_dir),
                convert_to_csv=False,
                dir_mode="zip"
            )
            print(f"Dataset uploaded successfully: {self.dataset_slug}")
        except Exception as e:
            print(f"Dataset upload failed: {e}")
            # Try to update existing dataset
            try:
                self.api.dataset_create_version(
                    folder=str(dataset_dir),
                    version_notes="Updated optimization pipeline",
                    convert_to_csv=False,
                    dir_mode="zip"
                )
                print(f"Dataset updated successfully: {self.dataset_slug}")
            except Exception as e2:
                print(f"Dataset update also failed: {e2}")
                raise
        
        return self.dataset_slug
    
    def create_execution_notebook(self) -> str:
        """
        Create a Kaggle notebook that executes the optimization.
        """
        print("Creating execution notebook...")
        
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# Multi-Objective GNN Hyperparameter Optimization\n",
                        "\n",
                        "This notebook runs comprehensive hyperparameter optimization for a Graph Neural Network pipeline on the ZINC dataset.\n",
                        "\n",
                        "## Objectives:\n",
                        "1. **Primary**: Minimize validation Mean Absolute Error\n",
                        "2. **Secondary**: Minimize memory usage, maximize throughput, minimize latency, minimize training time\n",
                        "\n",
                        "## Pipeline Components:\n",
                        "- Data Augmentation\n",
                        "- Dimensionality Reduction (PCA)\n",
                        "- Graph Neural Network (GCN/GIN/GAT/etc.)\n",
                        "- Graph Pooling (Mean/Max/Attention/Set2Set)\n",
                        "- Regression Head\n"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Install required packages\n",
                        "!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n",
                        "!pip install -q torch-geometric\n",
                        "!pip install -q pytorch-lightning\n",
                        "!pip install -q optuna\n",
                        "!pip install -q wandb\n",
                        "!pip install -q psutil\n",
                        "!pip install -q rdkit-pypi"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Setup environment\n",
                        "import os\n",
                        "import sys\n",
                        "import warnings\n",
                        "warnings.filterwarnings('ignore')\n",
                        "\n",
                        "# Add dataset path to Python path\n",
                        "sys.path.append('/kaggle/input/gnn-hyperopt-pipeline')\n",
                        "\n",
                        "# Verify GPU availability\n",
                        "import torch\n",
                        "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
                        "if torch.cuda.is_available():\n",
                        "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n",
                        "    print(f\"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\")"
                    ]
                },
                {
                    "cell_type": "code", 
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Configure Weights & Biases\n",
                        "import wandb\n",
                        "import os\n",
                        "\n",
                        "# Method 1: Use Kaggle Secrets (Recommended)\n",
                        "# Go to Account -> Secrets -> Add Secret\n",
                        "# Name: WANDB_API_KEY, Value: your W&B API key\n",
                        "try:\n",
                        "    from kaggle_secrets import UserSecretsClient\n",
                        "    user_secrets = UserSecretsClient()\n",
                        "    wandb_key = user_secrets.get_secret(\"WANDB_API_KEY\")\n",
                        "    wandb.login(key=wandb_key)\n",
                        "    print(\"✅ Weights & Biases authentication successful via Kaggle Secrets!\")\n",
                        "except Exception as e:\n",
                        "    print(f\"⚠️  W&B authentication via Kaggle Secrets failed: {e}\")\n",
                        "    print(\"Trying alternative methods...\")\n",
                        "    \n",
                        "    # Method 2: Set API key directly (less secure - remove after testing)\n",
                        "    # wandb_key = \"your-api-key-here\"  # Replace with your actual key\n",
                        "    # wandb.login(key=wandb_key)\n",
                        "    \n",
                        "    # Method 3: Run without W&B (fallback)\n",
                        "    print(\"Running without W&B logging...\")\n",
                        "    os.environ['WANDB_MODE'] = 'disabled'"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Import and run optimization\n",
                        "import sys\n",
                        "sys.path.append('/kaggle/input/gnn-hyperopt-pipeline')\n",
                        "\n",
                        "from optuna_multiobjective_optimization import main, OptimizationConfig\n",
                        "\n",
                        "# Run the optimization\n",
                        "print(\"Starting Multi-Objective Hyperparameter Optimization...\")\n",
                        "main()"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Analyze results\n",
                        "import pandas as pd\n",
                        "import matplotlib.pyplot as plt\n",
                        "import pickle\n",
                        "\n",
                        "# Load and display results\n",
                        "try:\n",
                        "    results_df = pd.read_csv('optimization_results.csv')\n",
                        "    print(\"Optimization Results Summary:\")\n",
                        "    print(f\"Total trials completed: {len(results_df)}\")\n",
                        "    print(f\"Best MAE: {results_df['val_mae'].min():.4f}\")\n",
                        "    print(f\"Best Memory Usage: {results_df['memory_gb'].min():.2f} GB\")\n",
                        "    print(f\"Best Throughput: {results_df['throughput_samples_sec'].max():.2f} samples/sec\")\n",
                        "    \n",
                        "    # Display top 10 trials by MAE\n",
                        "    print(\"\\nTop 10 trials by validation MAE:\")\n",
                        "    top_trials = results_df.nsmallest(10, 'val_mae')\n",
                        "    print(top_trials[['trial_number', 'val_mae', 'memory_gb', 'throughput_samples_sec']].to_string(index=False))\n",
                        "    \n",
                        "except FileNotFoundError:\n",
                        "    print(\"Results file not found. Optimization may have failed or is still running.\")"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10.12"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        # Save notebook
        notebook_dir = self.project_path / "kaggle_notebook"
        notebook_dir.mkdir(exist_ok=True)
        
        notebook_path = notebook_dir / "gnn-optimization.ipynb"
        with open(notebook_path, "w") as f:
            json.dump(notebook_content, f, indent=2)
        
        # Create kernel metadata
        kernel_metadata = {
            "id": self.kernel_slug,
            "title": "GNN Multi-Objective Hyperparameter Optimization",
            "code_file": "gnn-optimization.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [self.dataset_slug],
            "competition_sources": [],
            "kernel_sources": []
        }
        
        with open(notebook_dir / "kernel-metadata.json", "w") as f:
            json.dump(kernel_metadata, f, indent=2)
        
        print(f"Notebook created at {notebook_path}")
        return str(notebook_dir)
    
    def deploy_and_run(self, wandb_api_key: Optional[str] = None) -> str:
        """
        Complete deployment process: upload dataset, create notebook, and submit.
        """
        print("Starting Kaggle deployment process...")
        
        # Step 1: Upload dataset
        dataset_slug = self.prepare_dataset()
        
        # Step 2: Create and upload notebook
        notebook_dir = self.create_execution_notebook()
        
        # Step 3: Submit notebook
        try:
            self.api.kernels_push(notebook_dir)
            print(f"Notebook submitted successfully: {self.kernel_slug}")
            
            # Get submission URL
            kernel_url = f"https://www.kaggle.com/code/{self.kernel_slug}"
            print(f"Monitor execution at: {kernel_url}")
            
            return kernel_url
            
        except Exception as e:
            print(f"Notebook submission failed: {e}")
            raise
    
    def monitor_execution(self, check_interval: int = 300) -> Dict[str, Any]:
        """
        Monitor the execution status of the submitted kernel.
        
        Args:
            check_interval: Time in seconds between status checks
        """
        print(f"Monitoring kernel execution: {self.kernel_slug}")
        
        while True:
            try:
                status = self.api.kernels_status(self.kernel_slug)
                print(f"Current status: {status}")
                
                if status in ["complete", "failed", "cancelled"]:
                    break
                
                time.sleep(check_interval)
                
            except Exception as e:
                print(f"Error checking status: {e}")
                time.sleep(check_interval)
        
        # Download results if completed successfully
        if status == "complete":
            try:
                output_path = self.project_path / "kaggle_results"
                output_path.mkdir(exist_ok=True)
                
                self.api.kernels_output(self.kernel_slug, str(output_path))
                print(f"Results downloaded to: {output_path}")
                
                return {"status": "success", "output_path": str(output_path)}
                
            except Exception as e:
                print(f"Failed to download results: {e}")
                return {"status": "completed_no_download", "error": str(e)}
        
        return {"status": status}


def setup_kaggle_credentials():
    """
    Setup Kaggle API credentials.
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    credentials_path = kaggle_dir / "kaggle.json"
    
    if not credentials_path.exists():
        print("Kaggle credentials not found!")
        print("Please follow these steps:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Download the kaggle.json file")
        print(f"4. Place it at: {credentials_path}")
        return False
    
    # Set proper permissions
    credentials_path.chmod(0o600)
    return True


def main():
    """
    Main deployment script.
    """
    # Check Kaggle credentials
    if not setup_kaggle_credentials():
        return
    
    # Configuration
    project_path = os.getcwd()
    kaggle_username = input("Enter your Kaggle username: ").strip()
    
    if not kaggle_username:
        print("Kaggle username is required!")
        return
    
    # Ask for W&B API key (optional)
    wandb_key = input("Enter your Weights & Biases API key (optional, press Enter to skip): ").strip()
    
    # Create deployer and run
    deployer = KaggleDeployer(project_path, kaggle_username)
    
    try:
        kernel_url = deployer.deploy_and_run(wandb_key if wandb_key else None)
        print(f"\n🚀 Deployment successful!")
        print(f"📊 Monitor your optimization at: {kernel_url}")
        
        # Ask if user wants to monitor execution
        monitor = input("\nWould you like to monitor the execution? (y/n): ").strip().lower()
        
        if monitor == 'y':
            result = deployer.monitor_execution()
            print(f"\nExecution finished with status: {result['status']}")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")


if __name__ == "__main__":
    main()
