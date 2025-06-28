# 🚀 Kaggle Test Run - Step-by-Step Guide

Follow these exact steps to run your multi-objective GNN optimization on Kaggle:

## 📋 Pre-Requirements Checklist

### 1. Get Your W&B API Key
- Go to [wandb.ai/settings](https://wandb.ai/settings)
- Copy your API key (looks like: `1a2b3c4d5e6f7g8h9i0j...`)
- Keep it handy for step 3

### 2. (Optional) Setup Supabase Database
- Go to [supabase.com](https://supabase.com) and create a free project
- Note your project URL: `your-project.supabase.co`
- Note your database password from project settings
- Keep these handy for step 3

## 🔧 Kaggle Setup Steps

### Step 1: Upload Your Code to Kaggle

**Option A: Create Kaggle Dataset (Recommended)**
1. Zip your entire `kaggle_optuna_project` folder
2. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
3. Click "New Dataset"
4. Upload the zip file
5. Title: "GNN Multi-Objective Optimization Pipeline"
6. Make it Public or Private (your choice)
7. Click "Create"
8. **Note the dataset name** (e.g., `your-username/gnn-optimization`)

**Option B: Use GitHub (If your repo is public)**
- Just note your GitHub repo URL: `https://github.com/your-username/your-repo.git`

### Step 2: Create Kaggle Notebook
1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Choose "Notebook" (not Script)
4. **Important Settings:**
   - **Accelerator:** GPU T4 x2 (or whatever is available)
   - **Internet:** ON (required for W&B and Supabase)
   - **Language:** Python

### Step 3: Add Kaggle Secrets
1. In your new notebook, click "Add-ons" → "Secrets"
2. Add these secrets:

**Required:**
- **Name:** `WANDB_API_KEY`
- **Value:** Your W&B API key from step 1

**Optional (for persistent storage):**
- **Name:** `SUPABASE_URL`
- **Value:** `your-project.supabase.co` (no https://)
- **Name:** `SUPABASE_PASSWORD`
- **Value:** Your Supabase database password

### Step 4: Add Your Dataset (if using Option A)
1. In notebook settings, click "Input"
2. Search for your dataset name
3. Add it to the notebook

## 📝 Notebook Code

Copy and paste this code into your Kaggle notebook:

### Cell 1: Setup and Install
```python
# Install required packages
!pip install optuna wandb psycopg2-binary torch-geometric

# Check GPU
import torch
print(f"✅ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Cell 2: Setup Code
```python
import os
import sys
import shutil

# Method A: If you uploaded as Kaggle Dataset
dataset_path = "/kaggle/input/your-dataset-name"  # CHANGE THIS to your dataset name
if os.path.exists(dataset_path):
    shutil.copytree(dataset_path, "/kaggle/working/optimization", dirs_exist_ok=True)
    os.chdir("/kaggle/working/optimization")
    print("✅ Code copied from Kaggle Dataset")
else:
    # Method B: Clone from GitHub
    !git clone https://github.com/your-username/your-repo.git /kaggle/working/optimization
    os.chdir("/kaggle/working/optimization/kaggle_optuna_project")
    print("✅ Code cloned from GitHub")

sys.path.append(os.getcwd())
print(f"Working directory: {os.getcwd()}")
```

### Cell 3: Check Environment
```python
# Check secrets
wandb_key = os.getenv('WANDB_API_KEY')
supabase_url = os.getenv('SUPABASE_URL')
supabase_password = os.getenv('SUPABASE_PASSWORD')

print(f"W&B API Key: {'✅ Found' if wandb_key else '❌ Missing'}")
print(f"Supabase URL: {'✅ Found' if supabase_url else '⚠️ Optional'}")
print(f"Supabase Password: {'✅ Found' if supabase_password else '⚠️ Optional'}")

# Test imports
try:
    from optuna_multiobjective_optimization import OptimizationConfig, MultiObjectiveOptimizer
    from src.utils import setup_environment
    from src.pipeline_components import load_zinc_dataset
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    print("Check your file paths and structure")
```

### Cell 4: Test Dataset Loading
```python
# Test dataset
setup_environment()
try:
    train_dataset, val_dataset = load_zinc_dataset()
    print(f"✅ Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val")
except Exception as e:
    print(f"❌ Dataset error: {e}")
```

### Cell 5: Configure and Run Optimization
```python
# Configure for test run
config = OptimizationConfig(
    study_name="kaggle_test_run",
    n_trials=5,  # Small number for test run
    max_epochs=10,  # Fewer epochs for testing
    early_stopping_patience=3,
    wandb_project="gnn-test-kaggle",
    batch_size_options=[32, 64],  # Smaller batches for test
    memory_limit_gb=12.0,
    time_limit_hours=1.0,  # 1 hour for test run
)

# Setup Supabase if available
use_supabase = bool(supabase_url and supabase_password)
supabase_config = {
    'url': supabase_url,
    'password': supabase_password
} if use_supabase else None

print(f"Configuration:")
print(f"  Trials: {config.n_trials}")
print(f"  Max Epochs: {config.max_epochs}")
print(f"  Supabase: {'✅ Enabled' if use_supabase else '❌ Disabled'}")
```

### Cell 6: Run Optimization
```python
# Create and run optimizer
print("🚀 Starting optimization...")

try:
    optimizer = MultiObjectiveOptimizer(
        config=config,
        use_supabase=use_supabase,
        supabase_config=supabase_config
    )
    
    # Run optimization
    optimizer.run_optimization()
    print("✅ Optimization completed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
```

### Cell 7: Check Results
```python
# Check outputs
import pandas as pd
from pathlib import Path

# Check checkpoints
checkpoint_dir = Path("/kaggle/working/checkpoints")
if checkpoint_dir.exists():
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    print(f"🔖 Checkpoints: {len(checkpoints)} files")
    for ckpt in checkpoints:
        print(f"   {ckpt.name}")

# Check results CSV
if os.path.exists("optimization_results.csv"):
    df = pd.read_csv("optimization_results.csv")
    print(f"\n📊 Results: {len(df)} trials completed")
    if len(df) > 0:
        best_trial = df.loc[df['val_mae'].idxmin()]
        print(f"Best MAE: {best_trial['val_mae']:.4f}")
        print(f"Best Config: {best_trial['gnn_type']}, {best_trial['hidden_dim']} dims")

print(f"\n📁 All files in /kaggle/working/:")
!ls -la /kaggle/working/
```

## ⚠️ Important Notes

### Expected Runtime:
- **Test run (5 trials):** ~30-60 minutes
- **Full run (50 trials):** ~6-8 hours

### What to Monitor:
1. **Cell outputs** for progress updates
2. **W&B dashboard** (if API key provided): `wandb.ai/your-username/gnn-test-kaggle`
3. **Memory usage** in Kaggle's resource monitor
4. **GPU utilization** should be ~80-95%

### If Something Goes Wrong:
1. **Import errors:** Check file paths in Cell 2
2. **Dataset errors:** Verify your ACAgraphML package structure
3. **Memory errors:** Reduce batch_size_options to `[16, 32]`
4. **Time limits:** Kaggle has 9-hour notebook limit

## 🎯 Success Indicators

You'll know it's working when you see:
- ✅ GPU detected and used
- ✅ Dataset loaded successfully  
- ✅ W&B logging (if API key provided)
- ✅ Trial progress messages
- ✅ Checkpoints being saved
- ✅ CSV results file created

## 🚀 Ready to Launch!

Once you've:
1. ✅ Uploaded your code to Kaggle
2. ✅ Added W&B API key to secrets
3. ✅ Created notebook with GPU enabled
4. ✅ Copied the code above

Just run the cells in order and watch your multi-objective optimization in action!

**Good luck with your test run!** 🎉
