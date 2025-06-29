# 🚀 Kaggle Setup Guide: Multi-O## 🎯 Prerequisites

**Required:**

- **Kaggle Account**: With phone verification for GPU access
- **Weights & Biases Account**: Sign up at [wandb.ai](https://wandb.ai) (required for experiment tracking)

**Optional (for advanced features):**

- **Supabase Account**: Sign up at [supabase.com](https://supabase.com) (for persistent remote database storage) Graph Regression Optimization

This guide provides step-by-step instructions for running the multi-objective hyperparameter optimization notebook on Kaggle with GPU acceleration, Weights & Biases logging, and remote Optuna database storage.

## ✨ Quick Start Summary

**Minimum Required Setup:**

1. Set `WANDB_API_KEY` in Kaggle secrets (get from [wandb.ai](https://wandb.ai))
2. Enable GPU in notebook settings
3. Run all cells sequentially

**Optional Advanced Features:**

- Set `SUPABASE_DB_URL` for remote database storage (persistent across sessions)
- Set `WANDB_PROJECT` and `WANDB_ENTITY` for custom W&B organization

**Key Changes in This Version:**

- ✅ **Simplified secrets**: Only W&B API key required, others optional
- ✅ **Direct Kaggle integration**: Uses `kaggle_secrets` instead of environment variables
- ✅ **Complete Supabase URL**: Single connection string instead of separate components
- ✅ **Graceful fallbacks**: Works with minimal configuration, enhanced with full setup

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Kaggle Secrets Setup](#kaggle-secrets-setup)
3. [Dataset Upload](#dataset-upload)
4. [Notebook Configuration](#notebook-configuration)
5. [Running the Optimization](#running-the-optimization)
6. [Monitoring and Results](#monitoring-and-results)
7. [Troubleshooting](#troubleshooting)
8. [Expected Results](#expected-results)

## � Prerequisites

Before starting, ensure you have:

- **Kaggle Account**: With phone verification for GPU access
- **Weights & Biases Account**: Sign up at [wandb.ai](https://wandb.ai)
- **Supabase Account**: Sign up at [supabase.com](https://supabase.com) (for remote Optuna database)
- **Python Package**: Your `ACA_GraphML_Project` should be available on GitHub

## 🔐 Kaggle Secrets Setup

### Step 1: Create Required Accounts

#### Weights & Biases Setup (Required)

1. Go to [wandb.ai](https://wandb.ai) and sign up/login
2. Navigate to your profile settings
3. Go to "API Keys" section
4. Copy your API key (format: `xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`)

#### Supabase Database Setup (Optional - for remote storage)

**Note:** This is optional. Without Supabase, the notebook will use local SQLite storage, which works fine for most use cases.

1. Go to [supabase.com](https://supabase.com) and sign up/login
2. Create a new project (choose any name, e.g., "optuna-storage")
3. Wait for the project to be ready (2-3 minutes)
4. Go to **Settings** → **Database**
5. Scroll down to find the **"Connection string"** section
6. Copy the **URI connection string** (it should look like: `postgresql://postgres.abcdefgh:[YOUR-PASSWORD]@db.abcdefgh.supabase.co:5432/postgres`)
7. **Important:** Make sure to replace `[YOUR-PASSWORD]` with your actual database password

### Step 2: Configure Kaggle Secrets

1. Go to your [Kaggle account settings](https://www.kaggle.com/settings/account)
2. Scroll down to the **"Secrets"** section
3. Add the following secrets (click "Add Secret" for each):

| Secret Key        | Value                               | Required    | Description             |
| ----------------- | ----------------------------------- | ----------- | ----------------------- |
| `WANDB_API_KEY`   | Your W&B API key                    | ✅ Yes      | For experiment tracking |
| `WANDB_PROJECT`   | `zinc-graph-regression` (or custom) | ⚠️ Optional | W&B project name        |
| `WANDB_ENTITY`    | Your W&B username                   | ⚠️ Optional | Your W&B username/team  |
| `SUPABASE_DB_URL` | Complete PostgreSQL connection URL  | ⚠️ Optional | Remote database storage |

**Required Secret:**

```
WANDB_API_KEY: 1234567890abcdef1234567890abcdef12345678
```

**Optional Secrets (for advanced features):**

```
WANDB_PROJECT: zinc-graph-regression
WANDB_ENTITY: your-username
SUPABASE_DB_URL: postgresql://postgres.abcdefgh:your-password@db.abcdefgh.supabase.co:5432/postgres
```

**Important Notes:**

- Only `WANDB_API_KEY` is required for basic functionality
- `SUPABASE_DB_URL` should be the **complete PostgreSQL connection string** from Supabase
- If `WANDB_PROJECT` is not set, it defaults to "zinc-graph-regression"
- If `WANDB_ENTITY` is not set, W&B will use your default entity
- Without `SUPABASE_DB_URL`, the notebook will use local SQLite storage

## 📊 Dataset Upload

The notebook uses the ZINC dataset which will be automatically downloaded. However, if you want to use custom data:

1. **Create a Kaggle Dataset:**

   - Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
   - Click "New Dataset"
   - Upload your data files
   - Make it public or private as needed

2. **Add Dataset to Notebook:**
   - In your Kaggle notebook, click "Add Data"
   - Search for your dataset or the ZINC dataset
   - Add it to your notebook

## 📝 Notebook Configuration

### Step 1: Create New Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Choose **"Notebook"** (not script)
4. **Important**: Enable **GPU** acceleration:
   - Click "Settings" in the right panel
   - Under "Accelerator", select **"GPU P100"** or **"GPU T4 x2"**
   - This is crucial for performance!

### Step 2: Upload Notebook Content

1. **Option A: Copy-Paste**

   - Copy all cells from `KaggleNotebook.ipynb`
   - Paste into your Kaggle notebook

2. **Option B: Upload File**
   - Download the `KaggleNotebook.ipynb` from this repository
   - In Kaggle, use "File" → "Upload Notebook"

### Step 3: Verify Settings

Ensure your notebook has:

- ✅ **GPU enabled**
- ✅ **Internet enabled** (for package installation)
- ✅ **All secrets configured**

## 🚀 Running the Optimization

### Step 1: Run Initial Setup Cells

1. **Cell 1: Package Installation**

   ```python
   %pip install git+https://github.com/PaoloGinefra/ACA_GraphML_Project.git
   ```

   This should complete without errors (~2-3 minutes)

2. **Cell 2: Import Libraries**
   Verify all imports work and GPU is detected

3. **Cell 3: Configuration & Secrets**
   Check that secrets are loaded properly. Expected outputs:

   **With minimal configuration (W&B only):**

   ```
   ✅ Kaggle secrets interface available
   ✅ W&B API key loaded from Kaggle secrets
   ℹ️ Using default W&B project name (WANDB_PROJECT not set in secrets)
   ℹ️ W&B entity not set in secrets (will use default)
   ⚠️ Could not load SUPABASE_DB_URL from secrets
   ✅ W&B login successful
   ⚠️ Remote database not configured - will use local SQLite
   ```

   **With full configuration (W&B + Supabase):**

   ```
   ✅ Kaggle secrets interface available
   ✅ W&B API key loaded from Kaggle secrets
   ✅ W&B project name loaded from Kaggle secrets
   ✅ W&B entity loaded from Kaggle secrets
   ✅ Supabase database URL loaded from Kaggle secrets
   ✅ W&B login successful
   ✅ Optuna remote database configured
   ```

   **Configuration Summary:**
   At the end of Cell 3, you'll see a summary like this:

   ```
   🔧 Secrets Configuration Summary:
     Kaggle Secrets Available: ✅
     W&B API Key: ✅ Configured
     W&B Project: zinc-graph-regression
     W&B Entity: your-username
     Remote Database: ✅ Configured
     Study Name: zinc-graph-regression-multiobj
   ```

4. **Cell 4: Debug/Production Mode**
   Choose your optimization settings:
   - **DEBUG_MODE = True**: Quick test with 5 trials, 30 minutes
   - **DEBUG_MODE = False**: Full optimization with 50 trials, 6 hours
5. **Cell 5: System Monitor Setup**
   Should complete instantly

6. **Cell 6: Data Loading**
   Downloads and prepares ZINC dataset (~1-2 minutes)
   Expected output:
   ```
   📊 Dataset sizes:
     Training: 10000
     Validation: 1000
     Test: 1000
   ```

### Step 3: Optimization Setup

7. **Cell 7: Multi-Objective Setup**
   Defines the hyperparameter search space

8. **Cell 8: Objective Function**
   Sets up the 5-objective optimization function

9. **Cell 9: Create and Run Study**
   Defines the Optuna study creation and analysis functions

### Step 4: Run Optimization

10. **Cell 10: Main Execution**
    This is the main execution cell that:

- Creates the Optuna study
- Runs 50 trials (configurable)
- Saves results to CSV
- Evaluates the best model on test set

**Expected Runtime**: 3-6 hours (depending on GPU and configuration)

## 📈 Monitoring and Results

### During Execution

1. **Progress Monitoring:**

   - Watch the trial progress in notebook output
   - Each trial shows trial number and current objectives
   - Failed trials are logged and skipped

2. **W&B Dashboard:**

   - Go to [wandb.ai](https://wandb.ai)
   - Navigate to your project (`zinc-graph-regression`)
   - Monitor real-time metrics, hyperparameters, and system usage

3. **Memory/GPU Usage:**
   - Kaggle shows GPU utilization in the right panel
   - System metrics are logged in each trial

### Results Analysis

After completion, you'll get:

1. **Console Output:**

   ```
   🎉 Optimization completed!
   📊 Optimization Results Analysis
   Total trials: 50
   Completed trials: 45
   Pareto-optimal solutions: 8
   ```

2. **CSV Results:**

   - `optuna_results.csv` with all trial results
   - Contains hyperparameters and all objective values

3. **Best Model Evaluation:**
   ```
   🏆 Best model (lowest validation MAE):
   Trial: 23
   Validation MAE: 0.1234
   Test MAE: 0.1456
   Memory consumption: 256.7 MB
   Training time: 2.34 minutes
   ```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. Package Installation Fails

```bash
ERROR: Could not install packages due to an EnvironmentError
```

**Solution:**

- Restart notebook and try again
- Check if repository is public and accessible
- Try installing dependencies individually

#### 2. GPU Not Detected

```python
CUDA available: False
```

**Solution:**

- Go to Settings → Accelerator → Select "GPU"
- Restart notebook
- Ensure you have phone verification on Kaggle

#### 3. W&B Authentication Error

```
⚠️ W&B not configured - set WANDB_API_KEY in Kaggle secrets
```

**Solution:**

- Double-check secret name: `WANDB_API_KEY` (exact case)
- Regenerate API key from W&B dashboard
- Restart notebook after adding secrets

#### 4. Supabase Connection Error

```
⚠️ Could not load SUPABASE_DB_URL from secrets
⚠️ Remote database not configured - will use local SQLite
```

**Solution:**

- Verify the `SUPABASE_DB_URL` is the complete PostgreSQL connection string
- Format should be: `postgresql://postgres.project_id:password@db.project_id.supabase.co:5432/postgres`
- Ensure you replaced `[YOUR-PASSWORD]` with your actual database password
- **Note:** Local SQLite fallback still works fine for most use cases

#### 5. Kaggle Secrets Not Found

```
⚠️ Kaggle secrets not available - running outside Kaggle environment
```

**Solution:**

- Ensure you're running in a Kaggle notebook (not locally)
- Secrets are only available in Kaggle environment
- Check that secrets are properly configured in your Kaggle account settings

#### 6. CUDA Out of Memory

```bash
RuntimeError: CUDA out of memory
```

**Solution:**

- Reduce `BATCH_SIZE` from 32 to 16 or 8
- Reduce `num_trials` from 50 to 25
- Use smaller model configurations
- Restart notebook to clear GPU memory

#### 7. Kaggle Timeout

```
Your notebook session is about to expire
```

**Solution:**

- Reduce `timeout_hours` and `num_trials`
- Save intermediate results more frequently
- Consider running in multiple sessions

### Debug Mode

Add this cell at the beginning for debugging:

```python
# Debug mode - set smaller values for testing
DEBUG_MODE = True  # Set to False for full run

if DEBUG_MODE:
    CONFIG.update({
        'num_trials': 5,
        'timeout_hours': 0.5,
        'max_epochs': 10,
        'early_stopping_patience': 5,
    })
    print("🐛 DEBUG MODE: Using reduced settings")
```

## 📊 Expected Results

### Performance Benchmarks

Based on the ZINC dataset characteristics, you should expect:

#### Validation MAE Targets:

- **Baseline (random hyperparams)**: 0.8-1.2
- **Good optimization**: 0.3-0.6
- **Excellent optimization**: 0.2-0.4

#### System Metrics:

- **Memory consumption**: 100-1000 MB (depending on model size)
- **Training time per trial**: 1-10 minutes
- **Throughput**: 50-500 samples/second
- **Total optimization time**: 3-6 hours

#### Pareto Front Analysis:

The multi-objective optimization will find trade-offs between:

1. **Low MAE** (accuracy)
2. **Low memory** (efficiency)
3. **Fast training** (speed)
4. **High throughput** (scalability)
5. **Low latency** (responsiveness)

### Sample Results Table

| Trial | Val MAE | Memory (MB) | Time (min) | Throughput | Architecture   |
| ----- | ------- | ----------- | ---------- | ---------- | -------------- |
| 23    | 0.234   | 256         | 2.3        | 342        | GINEConv-128-4 |
| 45    | 0.267   | 128         | 1.8        | 458        | SAGE-64-3      |
| 12    | 0.298   | 512         | 4.1        | 198        | GAT-256-5      |

## 🎯 Next Steps

After successful completion:

1. **Analyze Results:**

   - Download `optuna_results.csv`
   - Analyze Pareto-optimal solutions
   - Identify best trade-offs for your use case

2. **Model Deployment:**

   - Retrain best model with full dataset
   - Export for production use
   - Set up monitoring pipeline

3. **Further Optimization:**
   - Extend search space with new hyperparameters
   - Add more objectives (e.g., inference latency)
   - Try ensemble methods with top models

## 📝 Tips for Success

1. **Start Small:** Run with `DEBUG_MODE=True` first to test everything works
2. **Monitor Resources:** Watch GPU memory and adjust batch sizes accordingly
3. **Save Progress:** Results are automatically saved to CSV and W&B
4. **Be Patient:** Full optimization takes 3-6 hours - don't interrupt
5. **Check Logs:** W&B provides detailed logging and visualization

## 🤝 Support

If you encounter issues:

1. Check this troubleshooting guide first
2. Verify all prerequisites are met
3. Try debug mode with reduced settings
4. Check Kaggle community forums
5. Open an issue in the project repository

## 📄 License

This guide is part of the ACA GraphML project. See the main project license for details.

---

**Happy optimizing! 🚀**

_Last updated: June 2025_

- Go to [wandb.ai](https://wandb.ai)
- Sign up for a free account
- Create a new project (e.g., "zinc-graph-regression")

2. **Get API Key**:
   - Go to your W&B settings
   - Copy your API key from the "API keys" section
   - Note your username/entity name

## 🔐 Kaggle Secrets Configuration

**IMPORTANT**: Store all credentials as Kaggle secrets for security.

1. **Navigate to Kaggle Secrets**:

   - Go to your Kaggle account settings
   - Click on "Secrets" in the left sidebar
   - Click "Add a new secret"

2. **Add the following secrets**:

   | Secret Name                 | Value                     | Description                    |
   | --------------------------- | ------------------------- | ------------------------------ |
   | `WANDB_API_KEY`             | `your_wandb_api_key`      | Your W&B API key               |
   | `WANDB_PROJECT`             | `zinc-graph-regression`   | W&B project name               |
   | `WANDB_ENTITY`              | `your_username`           | W&B username/entity            |
   | `SUPABASE_URL`              | `https://xyz.supabase.co` | Your Supabase project URL      |
   | `SUPABASE_SERVICE_ROLE_KEY` | `eyJ...`                  | Your Supabase service role key |

3. **Verify Secrets**:
   - Ensure all secrets are saved correctly
   - Secret names are case-sensitive

## 📂 Notebook Upload & Configuration

### 1. Upload the Notebook

1. **Create New Kaggle Notebook**:

   - Go to [kaggle.com/code](https://www.kaggle.com/code)
   - Click "New Notebook"
   - Choose "Notebook" type

2. **Upload Notebook File**:
   - Click "File" → "Upload Notebook"
   - Select the `KaggleNotebook.ipynb` file
   - Wait for upload to complete

### 2. Configure Notebook Settings

1. **Enable GPU**:

   - In the notebook, click Settings (gear icon) on the right
   - Under "Accelerator", select "GPU T4 x2" or "GPU P100"
   - Click "Save"

2. **Enable Internet**:

   - In Settings, turn ON "Internet"
   - This is required for downloading packages and connecting to external services

3. **Set Session Options**:
   - Choose "Always save version on run"
   - Set session timeout to maximum (9 hours for GPU)

### 3. Environment Variables

The notebook automatically reads from Kaggle secrets. No manual configuration needed if secrets are set correctly.

## 🚀 Running the Optimization

### 1. Pre-execution Checklist

- ✅ GPU enabled in notebook settings
- ✅ Internet enabled
- ✅ All secrets configured in Kaggle account
- ✅ W&B project created (if using W&B)
- ✅ Supabase project created (if using remote DB)

### 2. Execute the Notebook

1. **Run All Cells**:

   - Click "Run All" or use Ctrl+Shift+Enter
   - Monitor the progress in the output

2. **Expected Output Flow**:

   ```
   ✅ All imports successful!
   ✅ W&B login successful
   ✅ Optuna remote database configured
   ✅ System monitoring initialized
   ✅ Data preparation completed!
   🎯 Starting Multi-Objective Hyperparameter Optimization
   ```

3. **Monitor Progress**:
   - Watch for trial completions
   - Check memory usage warnings
   - Monitor validation MAE improvements

### 3. Expected Runtime

- **Per Trial**: 2-5 minutes (depends on model complexity)
- **Total Runtime**: 2-6 hours (for 50 trials)
- **Early Stopping**: Trials may be pruned early for efficiency

## 📊 Monitoring & Results

### 1. Real-time Monitoring

**In Notebook Output**:

- Trial progress and metrics
- Memory consumption tracking
- System performance metrics
- Pareto front analysis

**Weights & Biases Dashboard** (if configured):

- Real-time loss curves
- Hyperparameter importance
- System metrics tracking
- Cross-trial comparisons

**Optuna Dashboard** (if using remote DB):

- Multi-objective optimization plots
- Parameter importance analysis
- Trial history and pruning

### 2. Results Files

The notebook generates several output files:

| File                 | Description                             |
| -------------------- | --------------------------------------- |
| `optuna_results.csv` | Complete trial results with all metrics |
| `optuna_study.db`    | Local SQLite database (if no remote DB) |
| Kaggle output logs   | Detailed execution logs                 |

### 3. Key Metrics to Monitor

1. **Validation MAE**: Primary objective (lower is better)
2. **Memory Usage**: Peak GPU/RAM consumption
3. **Training Time**: Time to convergence
4. **Throughput**: Samples/second processed
5. **Model Parameters**: Model complexity

## 🎯 Interpreting Results

### 1. Pareto Front Analysis

The notebook automatically identifies Pareto-optimal solutions - models that represent the best trade-offs between objectives.

**Look for**:

- Models with low MAE and reasonable resource usage
- Memory-efficient models for deployment
- Fast-training models for rapid iteration

### 2. Best Model Selection

The notebook evaluates the best model (lowest validation MAE) on the test set:

```
🏆 Best model (lowest validation MAE):
Trial: 23
Validation MAE: 0.1234
Memory consumption: 512.3 MB
Training time: 3.45 minutes

🎯 Final Test Results:
Test MAE: 0.1245
Model parameters: 1,234,567
```

### 3. Hyperparameter Insights

Check the results for patterns:

- Which architectures perform best?
- How does model size affect performance?
- What are the optimal regularization settings?

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory Error**:

   ```
   Solution: Reduce batch size or model complexity
   - Decrease BATCH_SIZE in config
   - Try smaller hidden dimensions
   - Reduce number of layers
   ```

2. **W&B Authentication Failed**:

   ```
   Check: WANDB_API_KEY secret is correct
   Verify: W&B project exists
   Ensure: Internet is enabled in notebook
   ```

3. **Supabase Connection Failed**:

   ```
   Check: SUPABASE_URL format (https://xyz.supabase.co)
   Verify: Service role key is correct
   Fallback: Uses local SQLite automatically
   ```

4. **Package Installation Errors**:
   ```
   Solution: The first cell installs the package from GitHub
   Wait: Installation may take 2-3 minutes
   Restart: Kernel restart may be needed after installation
   ```

### Performance Optimization

1. **Speed up optimization**:

   - Reduce `num_trials` in CONFIG
   - Decrease `max_epochs`
   - Use smaller dataset subset

2. **Memory optimization**:

   - Reduce `BATCH_SIZE`
   - Limit model complexity in hyperparameter search
   - Enable gradient checkpointing

3. **Remote storage issues**:
   - Falls back to local SQLite automatically
   - Local storage works but doesn't persist across sessions

## 📈 Advanced Configuration

### Custom Hyperparameter Spaces

Modify the `create_model_with_config` function to add/remove hyperparameters:

```python
# Add new hyperparameter
num_heads = trial.suggest_int('num_heads', 1, 8)  # For attention models

# Modify search space
hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])  # Smaller range
```

### Objective Weights

To prioritize certain objectives, modify the NSGA-II sampler or implement custom selection logic in the analysis phase.

### Extended Runtime

For longer optimization runs:

1. Increase `timeout_hours` in CONFIG
2. Use Kaggle's premium GPU for longer sessions
3. Implement checkpointing for resume capability

## 🔄 Continuing Optimization

To continue an optimization in a new session:

1. **With Remote Database**:

   - Study automatically resumes from Supabase
   - Previous trials are preserved

2. **With Local Database**:
   - Download `optuna_study.db` from previous session
   - Upload to new session before running

## 📋 Expected Results

Based on the ZINC dataset and your pipeline:

- **Baseline MAE**: ~0.8-1.0 (simple models)
- **Good Performance**: ~0.3-0.5 MAE
- **Excellent Performance**: ~0.15-0.25 MAE
- **Memory Usage**: 200MB - 2GB (depending on model)
- **Training Time**: 1-10 minutes per trial

## 🎉 Conclusion

This setup provides a comprehensive multi-objective optimization pipeline for graph neural networks with:

- ✅ Professional experiment tracking
- ✅ Remote database storage
- ✅ Comprehensive system monitoring
- ✅ Automated result analysis
- ✅ Production-ready model selection

The notebook handles failures gracefully and provides detailed logging for debugging. Results can be used for model deployment, academic research, or further optimization.

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all secrets are configured correctly
3. Ensure GPU and Internet are enabled
4. Check Kaggle quota limits
5. Review the notebook output for specific error messages

The system is designed to be robust and will fallback to local storage and CPU if needed, though performance will be reduced.
