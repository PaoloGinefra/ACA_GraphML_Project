# Multi-Objective GNN Hyperparameter Optimization with Optuna and Weights & Biases

This project implements a comprehensive **multi-objective hyperparameter optimization** system for Graph Neural Networks using your existing ACAgraphML pipeline. The system optimizes for multiple objectives simultaneously and provides remote experiment tracking.

## 🎯 What This System Does

### Multi-Objective Optimization
The system optimizes **5 objectives simultaneously**:

1. **Validation Mean Absolute Error (Primary)** - Minimize prediction error
2. **Memory Usage** - Minimize GPU/CPU memory consumption
3. **Throughput** - Maximize training/inference speed (samples/sec)
4. **Latency** - Minimize inference time per sample
5. **Training Time** - Minimize total training duration

### Why Multi-Objective?
Real-world ML models need to balance accuracy with computational constraints:
- **Memory**: Critical for deployment on resource-constrained devices
- **Speed**: Important for real-time applications and cost optimization
- **Training Time**: Affects research iteration speed and cloud costs

### Pipeline Components Optimized
- **Data Augmentation**: Self-loops, feature normalization, random walk positional encoding
- **Dimensionality Reduction**: PCA with configurable variance retention
- **GNN Architecture**: Layer type, depth, width, regularization
- **Graph Pooling**: Mean, max, attention, Set2Set pooling strategies
- **Regression Head**: Linear, MLP, deep MLP with various activations
- **Training Setup**: Optimizers, schedulers, learning rates, regularization

## 🏗️ Architecture Overview

```
Data → Augmentation → Dim Reduction → GNN → Pooling → Regressor → Prediction
  ↓           ↓            ↓         ↓       ↓         ↓
Optuna Hyperparameter Search Space (Multi-Objective)
  ↓
Weights & Biases Remote Logging
  ↓  
Kaggle GPU Environment
```

## 📁 Project Structure

```
kaggle_optuna_project/
├── optuna_multiobjective_optimization.py  # Main optimization script
├── deploy_to_kaggle.py                     # Kaggle deployment automation
├── setup_kaggle_project.py                # Setup script
├── requirements.txt                        # Dependencies
├── README.md                              # This file
├── src/
│   ├── pipeline_components.py             # Your ACAgraphML integration
│   ├── performance_monitor.py             # Resource monitoring
│   ├── utils.py                          # Utilities
│   └── ACAgraphML/                       # Your copied package
├── configs/
│   └── optimization_config.yaml          # Configuration file
└── data/                                  # Dataset directory
```

## 🚀 Quick Start

### 1. Setup the Project

```bash
cd kaggle_optuna_project
python setup_kaggle_project.py
```

This will:
- Copy your ACAgraphML package
- Update import statements for Kaggle compatibility
- Create comprehensive requirements file
- Verify all files are in place

### 2. Configure Weights & Biases

```bash
# Install wandb if not already installed
pip install wandb

# Login to your account
wandb login
```

Or get your API key from: https://wandb.ai/authorize

### 3. Setup Kaggle API

1. Go to https://www.kaggle.com/account
2. Click "Create New API Token"
3. Download `kaggle.json`
4. Place it at `~/.kaggle/kaggle.json` (Windows: `C:\Users\{username}\.kaggle\kaggle.json`)

### 4. Deploy to Kaggle

```bash
python deploy_to_kaggle.py
```

This will:
- Upload your code as a Kaggle dataset
- Create and submit a Kaggle notebook
- Start the optimization automatically
- Provide monitoring URL

## ⚙️ Configuration

### Optimization Parameters

Edit `configs/optimization_config.yaml` to customize:

```yaml
optimization:
  n_trials: 50                    # Number of optimization trials
  max_epochs: 30                  # Max epochs per trial
  early_stopping_patience: 5      # Early stopping patience
  memory_limit_gb: 12.0          # Memory limit (Kaggle: 12GB)
  time_limit_hours: 9.0          # Time limit (Kaggle: 9 hours)

wandb:
  project: "gnn-hyperopt-zinc"    # Your W&B project name
  entity: "your-username"         # Your W&B username
```

### Hyperparameter Search Spaces

The system searches over:

**GNN Architecture:**
- Types: GCN, GIN, GINE, GraphSAGE, GAT
- Hidden dimensions: 64, 128, 256, 512
- Layers: 2-6
- Dropout: 0.0-0.5
- Residual connections, layer normalization

**Pooling:**
- Types: mean, max, add, attention, set2set
- Processing steps for attention-based pooling

**Training:**
- Optimizers: Adam, AdamW, SGD
- Learning rates: 1e-5 to 1e-1 (log scale)
- Schedulers: cosine, plateau, step, none
- Weight decay: 1e-6 to 1e-2

## 📊 Monitoring and Results

### Real-time Monitoring

1. **Weights & Biases Dashboard**: Track all experiments in real-time
   - Hyperparameters for each trial
   - Training curves and validation metrics
   - System resource usage
   - Model architectures

2. **Kaggle Notebook**: Monitor execution status
   - GPU utilization
   - Memory usage
   - Training progress

### Results Analysis

After optimization completes, you'll get:

1. **`optimization_results.csv`**: Detailed results for all trials
2. **`optuna_study.pkl`**: Complete Optuna study object
3. **Pareto Front Analysis**: Best trade-offs between objectives
4. **Best Models**: Top performers for each objective

## 🔍 Understanding Multi-Objective Results

### Pareto Front
The system finds models that represent optimal trade-offs. For example:
- **Model A**: Lowest MAE but high memory usage
- **Model B**: Balanced MAE and memory with high throughput
- **Model C**: Fastest inference but slightly higher MAE

### Choosing the Best Model
Consider your deployment constraints:
- **Research**: Prioritize lowest MAE
- **Mobile/Edge**: Prioritize low memory and fast inference
- **Production Server**: Balance all objectives
- **Real-time Applications**: Prioritize throughput and latency

## 🛠️ Customization

### Adding New Objectives

To add a new objective (e.g., model size):

1. **Modify the objective function** in `optuna_multiobjective_optimization.py`:
```python
def objective_function(self, trial):
    # ... existing code ...
    model_size_mb = pipeline.get_model_size_mb()
    
    return val_mae, memory_usage_gb, throughput, latency, training_time, model_size_mb
```

2. **Update the study creation**:
```python
self.study = optuna.create_study(
    directions=["minimize", "minimize", "maximize", "minimize", "minimize", "minimize"]
)
```

### Using Different Datasets

To use a different dataset:

1. **Modify `load_zinc_dataset()`** in `src/pipeline_components.py`
2. **Update node/edge feature dimensions** in `create_pipeline_from_config()`
3. **Adjust hyperparameter ranges** in the configuration

### Custom Hyperparameter Spaces

Add new hyperparameters in `define_hyperparameter_space()`:

```python
# Example: Add L2 regularization
l2_reg = trial.suggest_float('l2_regularization', 1e-6, 1e-2, log=True)
```

## 🎓 Learning Resources

### Optuna Multi-Objective Optimization
- [Optuna Tutorial](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_multi_objective.html)
- [Multi-Objective Best Practices](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html)

### Graph Neural Networks
- [PyTorch Geometric Tutorials](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html)
- [Graph Neural Network Course](https://web.stanford.edu/class/cs224w/)

### MLOps and Experiment Tracking
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [PyTorch Lightning Guide](https://pytorch-lightning.readthedocs.io/)

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**: Run `setup_kaggle_project.py` to ensure all files are copied
2. **Kaggle Authentication**: Verify `kaggle.json` is in the correct location
3. **W&B Login**: Use `wandb login` or set API key in Kaggle secrets
4. **Memory Issues**: Reduce batch size or model dimensions in config
5. **Time Limits**: Reduce `n_trials` or `max_epochs` for Kaggle's 9-hour limit

### Performance Optimization

1. **Reduce Search Space**: Start with fewer hyperparameter options
2. **Early Stopping**: Use aggressive early stopping for faster trials
3. **Batch Size**: Optimize batch size for your GPU memory
4. **Parallel Trials**: Use multiple Kaggle notebooks for parallel search

## 📈 Expected Results

### ZINC Dataset Benchmarks
- **Baseline MAE**: ~0.5-0.8
- **State-of-the-art**: ~0.1-0.3
- **Memory Usage**: 2-8 GB depending on architecture
- **Training Time**: 5-30 minutes per epoch

### Optimization Outcomes
After 50 trials, expect to find:
- 3-5 models on the Pareto front
- 20-30% improvement over random search
- Clear trade-offs between objectives
- Insights into architecture importance

## 🤝 Contributing

This system is designed to be extensible. Consider contributing:
- New objective functions
- Additional GNN architectures
- Better hyperparameter search strategies
- Performance optimizations

## 📄 License

This project extends your ACAgraphML package and follows the same licensing terms.

---

**Ready to optimize?** Run `python setup_kaggle_project.py` to get started! 🚀
