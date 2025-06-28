# Kaggle Deployment Guide

This guide addresses your specific requirements for deploying the multi-objective GNN optimization pipeline on Kaggle.

## 🔑 Key Configuration Updates Made

### 1. Kaggle GPU Support ✅
The pipeline is now fully configured for Kaggle GPU environments:

```python
trainer = pl.Trainer(
    devices=1,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    precision='16-mixed' if torch.cuda.is_available() else 32,  # Mixed precision for Kaggle GPUs
    gradient_clip_val=1.0,
    strategy='auto',
)
```

**Features:**
- Automatic GPU detection
- Mixed precision training (16-bit) for faster training and lower memory usage
- Gradient clipping for training stability
- Optimized for single GPU (Kaggle standard)

### 2. ZINC Dataset Train/Val Splits ✅
Your existing ZINC dataset splits are now properly used:

```python
def load_zinc_dataset():
    train_dataset = ZINC_Dataset.loadDatasetZINC(split='train', subset=True, transform=oneHotTransform)
    val_dataset = ZINC_Dataset.loadDatasetZINC(split='val', subset=True, transform=oneHotTransform)
    return train_dataset, val_dataset
```

**No manual splitting** - uses your class's built-in train/validation splits directly.

### 3. Weights & Biases Authentication for Kaggle ✅

**Setup:**
1. Get your W&B API key from [wandb.ai/settings](https://wandb.ai/settings)
2. In Kaggle, go to your account settings → "Secrets"
3. Add a new secret:
   - **Name:** `WANDB_API_KEY`
   - **Value:** Your actual API key

**How it works:**
- The script automatically detects the `WANDB_API_KEY` environment variable
- Falls back to offline mode if authentication fails
- All experiments are logged to your W&B project

### 4. Model Checkpoint Storage ✅

**Kaggle Environment:**
```
Checkpoints saved to: /kaggle/working/checkpoints/
```

**Local Environment:**
```
Checkpoints saved to: ./checkpoints/
```

**Features:**
- Automatic directory creation
- Best model saving based on validation MAE
- Last checkpoint saving for resuming
- Kaggle-compatible paths for permanent storage

### 5. Supabase Database Integration ✅

**Setup Instructions:**

1. **Create Supabase Project:**
   - Go to [supabase.com](https://supabase.com)
   - Create a new project
   - Note your project URL and database password

2. **Setup Database (Optional - Optuna will auto-create tables):**
   ```sql
   -- Optuna will automatically create these tables, but you can check:
   -- trials, studies, trial_params, trial_values, etc.
   ```

3. **Configure Kaggle Secrets:**
   - **Secret Name:** `SUPABASE_URL`
   - **Secret Value:** `your-project.supabase.co` (without https://)
   - **Secret Name:** `SUPABASE_PASSWORD`  
   - **Secret Value:** Your database password

**Benefits:**
- Persistent study storage across multiple Kaggle runs
- Resume optimization from previous trials
- Access results from anywhere
- Automatic backup of all optimization data

## 🚀 Deployment Steps

### 1. Prepare Kaggle Notebook

```python
# Install required packages
!pip install optuna wandb psycopg2-binary supabase

# Clone your repository (if not using Kaggle Datasets)
!git clone https://github.com/your-username/your-repo.git
%cd your-repo/kaggle_optuna_project

# Run optimization
!python optuna_multiobjective_optimization.py
```

### 2. Alternative: Use Kaggle Dataset

1. Upload your code as a Kaggle Dataset
2. Create notebook and add the dataset
3. Import and run:

```python
import sys
sys.path.append('/kaggle/input/your-dataset-name')

from optuna_multiobjective_optimization import main
main()
```

### 3. Monitor Progress

**W&B Dashboard:**
- Real-time training metrics
- Hyperparameter tracking
- Model performance comparison

**Supabase Database:**
- Persistent trial storage
- Multi-run analysis
- Export capabilities

## 📊 Expected Output

### Checkpoints Location:
```
/kaggle/working/checkpoints/
├── trial_0_best_mae=0.1234.ckpt
├── trial_1_best_mae=0.1156.ckpt
├── last.ckpt
└── ...
```

### Results Files:
```
/kaggle/working/
├── optimization_results.csv    # Detailed trial results
├── optuna_study.pkl           # Complete study object
└── wandb/                     # Local W&B logs (if offline)
```

## 🔧 Configuration Options

### Basic Configuration:
```python
config = OptimizationConfig(
    n_trials=50,              # Adjust for available time
    max_epochs=30,            # Kaggle time limits
    early_stopping_patience=5,
    wandb_project="gnn-hyperopt-zinc",
)
```

### Resource Limits:
```python
config = OptimizationConfig(
    memory_limit_gb=12.0,     # Kaggle GPU memory
    time_limit_hours=9.0,     # Kaggle time limit
    batch_size_options=[16, 32, 64, 128],  # GPU-appropriate sizes
)
```

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Kaggle GPU    │    │   Supabase DB    │    │   W&B Cloud     │
│                 │    │                  │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │ Optuna      │ │◄──►│ │ Study Storage│ │    │ │ Experiment  │ │
│ │ Optimizer   │ │    │ │              │ │    │ │ Tracking    │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ └─────────────┘ │
│                 │    │                  │    │                 │
│ ┌─────────────┐ │    │                  │    │                 │
│ │ GNN         │ │    │                  │    │                 │
│ │ Training    │ │    │                  │    │                 │
│ └─────────────┘ │    │                  │    │                 │
│                 │    │                  │    │                 │
│ ┌─────────────┐ │    │                  │    │                 │
│ │ Checkpoints │ │    │                  │    │                 │
│ │ /kaggle/    │ │    │                  │    │                 │
│ │ working/    │ │    │                  │    │                 │
│ └─────────────┘ │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚦 Quick Test

Run the test script to verify everything works:

```bash
python test_local_optimization.py  # Test locally first
```

Then deploy to Kaggle with confidence!

## ⚠️ Important Notes

1. **GPU Memory:** Kaggle provides ~16GB GPU memory - the pipeline uses mixed precision to optimize usage
2. **Time Limits:** Kaggle has 9-hour limits - configure `n_trials` accordingly  
3. **Checkpoints:** Saved to `/kaggle/working/` for persistence between sessions
4. **Supabase:** Optional but recommended for long-term study persistence
5. **W&B:** Requires API key in Kaggle Secrets for cloud logging

All your requirements are now fully addressed! 🎉
