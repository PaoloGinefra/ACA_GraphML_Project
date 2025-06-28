# =============================================================================
# KAGGLE NOTEBOOK: Multi-Objective GNN Optimization Test Run
# Copy and paste each cell below into your Kaggle notebook
# =============================================================================

# CELL 1: Install Dependencies
# ----------------------------------------------------------------------------
!pip install optuna wandb psycopg2-binary torch-geometric

import torch
print(f"✅ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# CELL 2: Setup Code Directory
# ----------------------------------------------------------------------------
import os
import sys
import shutil

# OPTION A: If you uploaded as Kaggle Dataset
# Change "your-dataset-name" to your actual dataset name
dataset_path = "/kaggle/input/your-dataset-name"  # <-- CHANGE THIS

if os.path.exists(dataset_path):
    shutil.copytree(dataset_path, "/kaggle/working/optimization", dirs_exist_ok=True)
    os.chdir("/kaggle/working/optimization")
    print("✅ Code copied from Kaggle Dataset")
else:
    # OPTION B: Clone from GitHub (replace with your repo)
    !git clone https://github.com/your-username/your-repo.git /kaggle/working/optimization
    os.chdir("/kaggle/working/optimization/kaggle_optuna_project")
    print("✅ Code cloned from GitHub")

sys.path.append(os.getcwd())
print(f"Working directory: {os.getcwd()}")
!ls -la

# CELL 3: Check Environment and Secrets
# ----------------------------------------------------------------------------
# Check Kaggle Secrets
wandb_key = os.getenv('WANDB_API_KEY')
supabase_url = os.getenv('SUPABASE_URL')
supabase_password = os.getenv('SUPABASE_PASSWORD')

print("🔑 Kaggle Secrets Status:")
print(f"  W&B API Key: {'✅ Found' if wandb_key else '❌ Missing (add to secrets)'}")
print(f"  Supabase URL: {'✅ Found' if supabase_url else '⚠️ Optional'}")
print(f"  Supabase Password: {'✅ Found' if supabase_password else '⚠️ Optional'}")

# Test imports
print("\n📦 Testing imports...")
try:
    from optuna_multiobjective_optimization import OptimizationConfig, MultiObjectiveOptimizer
    from src.utils import setup_environment
    from src.pipeline_components import load_zinc_dataset
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    print("Check your file structure - should see 'src/' folder and main script")

# CELL 4: Test Dataset Loading
# ----------------------------------------------------------------------------
print("🧪 Testing dataset loading...")
setup_environment()

try:
    train_dataset, val_dataset = load_zinc_dataset()
    print(f"✅ Dataset loaded successfully:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Validation: {len(val_dataset)} samples")
except Exception as e:
    print(f"❌ Dataset loading failed: {e}")
    print("Make sure your ACAgraphML package is properly included")

# CELL 5: Configure Test Run
# ----------------------------------------------------------------------------
# Quick test configuration
config = OptimizationConfig(
    study_name="kaggle_test_run",
    n_trials=3,  # Small number for testing
    max_epochs=5,  # Quick epochs for testing
    early_stopping_patience=2,
    wandb_project="gnn-test-kaggle",
    batch_size_options=[32, 64],  # GPU-friendly sizes
    memory_limit_gb=12.0,
    time_limit_hours=0.5,  # 30 minutes max for test
)

# Supabase configuration
use_supabase = bool(supabase_url and supabase_password)
supabase_config = {
    'url': supabase_url,
    'password': supabase_password
} if use_supabase else None

print("⚙️ Test Configuration:")
print(f"   Trials: {config.n_trials}")
print(f"   Max Epochs: {config.max_epochs}")
print(f"   Time Limit: {config.time_limit_hours} hours")
print(f"   W&B Project: {config.wandb_project}")
print(f"   Supabase: {'✅ Enabled' if use_supabase else '❌ Disabled'}")

# CELL 6: Run Optimization
# ----------------------------------------------------------------------------
print("🚀 Starting Multi-Objective Optimization Test...")
print("This will optimize for: MAE, Memory, Throughput, Latency, Training Time")
print("=" * 60)

try:
    # Create optimizer
    optimizer = MultiObjectiveOptimizer(
        config=config,
        use_supabase=use_supabase,
        supabase_config=supabase_config
    )
    
    print(f"📊 Loaded dataset: {len(optimizer.train_dataset)} train, {len(optimizer.val_dataset)} val")
    
    # Run optimization
    optimizer.run_optimization()
    
    print("\n🎉 Optimization completed successfully!")
    
except KeyboardInterrupt:
    print("\n⚠️ Optimization interrupted by user")
except Exception as e:
    print(f"\n❌ Optimization failed: {e}")
    import traceback
    traceback.print_exc()

# CELL 7: Check Results
# ----------------------------------------------------------------------------
import pandas as pd
from pathlib import Path

print("📊 Checking Results...")
print("=" * 40)

# Check checkpoints
checkpoint_dir = Path("/kaggle/working/checkpoints")
if checkpoint_dir.exists():
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    print(f"\n🔖 Model Checkpoints: {len(checkpoints)} files")
    for ckpt in sorted(checkpoints):
        size_mb = ckpt.stat().st_size / (1024*1024)
        print(f"   {ckpt.name} ({size_mb:.1f} MB)")
else:
    print("\n❌ No checkpoints directory found")

# Check results
if os.path.exists("optimization_results.csv"):
    df = pd.read_csv("optimization_results.csv")
    print(f"\n📈 Optimization Results: {len(df)} trials completed")
    
    if len(df) > 0:
        # Show best trial
        best_idx = df['val_mae'].idxmin()
        best_trial = df.loc[best_idx]
        
        print(f"\n🏆 Best Trial (#{best_trial['trial_number']}):")
        print(f"   MAE: {best_trial['val_mae']:.4f}")
        print(f"   Model: {best_trial['gnn_type']}")
        print(f"   Hidden Dims: {best_trial['hidden_dim']}")
        print(f"   Memory: {best_trial['memory_gb']:.2f} GB")
        print(f"   Throughput: {best_trial['throughput_samples_sec']:.1f} samples/sec")
        
        # Show all trials
        print(f"\n📋 All Trials Summary:")
        summary = df[['trial_number', 'val_mae', 'memory_gb', 'gnn_type', 'hidden_dim']].round(4)
        print(summary.to_string(index=False))
else:
    print("\n❌ No results CSV found")

# Check W&B logs
if wandb_key:
    print(f"\n🔗 View detailed results at: https://wandb.ai/your-username/{config.wandb_project}")
else:
    print(f"\n⚠️ Add WANDB_API_KEY to Kaggle Secrets for online logging")

# List all output files
print(f"\n📁 All Output Files:")
!ls -la /kaggle/working/

print(f"\n✅ Test run complete! Check the results above.")

# =============================================================================
# END OF KAGGLE NOTEBOOK
# =============================================================================
