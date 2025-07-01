# ACA GraphML Project

A comprehensive Graph Machine Learning framework for molecular property prediction, developed as part of the Advanced Computer Architecture course at Politecnico di Milano. This project tackles graph regression on the ZINC dataset through extensive hyperparameter optimization and MLOps infrastructure.

For an overview of the project, see the [Project Report](./ACA_GraphMLProjectReport_PaoloGinefra.pdf)

## 🎯 Project Overview

This project implements a complete graph machine learning pipeline with the following key achievements:

1. **Comprehensive GML Package**: Built and extensively tested a Python package with highly parametrized graph ML pipeline supporting multiple GNN architectures, pooling strategies, and regression heads
2. **Dataset Analysis**: Comprehensive exploration and analysis of the ZINC molecular dataset
3. **Feature Engineering**: Pure feature engineering approach to molecular property prediction
4. **MLOps Infrastructure**: Advanced hyperparameter optimization system using Optuna with remote database logging and Weights & Biases integration for systematic investigation of optimal configurations

## 🚀 Quick Start

### Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/PaoloGinefra/ACA_GraphML_Project.git
```

or by downloading the repo and installing in editable mode:

```bash
git clone https://github.com/PaoloGinefra/ACA_GraphML_Project.git
cd ACA_GraphML_Project
pip install -e .
```

### Basic Usage

```python
from ACAgraphML import GDLPipeline, GNNConfig, PoolingConfig, RegressorConfig

# Create a standard pipeline for ZINC dataset
pipeline = GDLPipeline(
    node_features=28,
    edge_features=4,
    gnn_config=GNNConfig(hidden_dim=128, num_layers=4, layer_name="GINEConv"),
    pooling_config=PoolingConfig(pooling_type="attentional"),
    regressor_config=RegressorConfig(regressor_type="mlp", hidden_dims=[128, 64])
)

# For PyTorch Lightning integration
from ACAgraphML.Pipeline.LightningModules import GDLPipelineLightningModule

model = GDLPipelineLightningModule(
    node_features=28,
    edge_features=4,
    pipeline_config="standard",
    lr=1e-3
)
```

## 📦 Package Components

### Core Pipeline

- **GDLPipeline**: Main pipeline combining GNN + Pooling + Regressor
- **GNNModel**: Support for 13+ GNN architectures (GINEConv, GAT, GATv2, SAGE, PNA, etc.)
- **Pooling**: Multiple strategies (mean, max, attentional, Set2Set)
- **Regressor**: Various architectures (linear, MLP, residual, attention, ensemble)

### Training & Optimization

- **Lightning Module**: PyTorch Lightning wrapper with advanced training features
- **Hyperparameter Optimization**: Optuna integration with multi-objective support
- **Target Normalization**: Automatic target preprocessing and denormalization

### Data & Transforms

- **ZINC Dataset**: Ready-to-use molecular dataset loading
- **Transforms**: Feature engineering (OneHot encoding, PCA, augmentation)
- **Plotting**: Visualization utilities for graphs and results

### Utilities

- **Model Analysis**: Parameter counting, memory usage, performance metrics
- **Testing**: Comprehensive test suite (unit, integration, benchmark)

## 🧪 Running Tests

The project includes a comprehensive test suite covering all components:

```bash
# Run all tests
python tests/run_gdl_pipeline_tests.py all

# Quick tests (no benchmarks)
python tests/run_gdl_pipeline_tests.py quick

# Specific test types
python tests/run_gdl_pipeline_tests.py unit
python tests/run_gdl_pipeline_tests.py integration
python tests/run_gdl_pipeline_tests.py benchmark

# ZINC dataset specific tests
python tests/run_gdl_pipeline_tests.py zinc
```

Alternative pytest approach:

```bash
# Run specific test files
python -m pytest tests/test_GDLPipeline.py -v
python -m pytest tests/test_GNNmodel.py -v
python -m pytest tests/test_GDLPipelineLightningModule.py -v

# Run all tests with coverage
python -m pytest tests/ --cov=src/ACAgraphML --cov-report=html
```

## 🔧 Hyperparameter Optimization

The project includes a sophisticated MLOps infrastructure for hyperparameter optimization using Optuna and Weights & Biases.

### Setup Requirements

1. **Weights & Biases Account**: Sign up at [wandb.ai](https://wandb.ai)
2. **Optional Remote Database**: Supabase or PostgreSQL for persistent storage

### Running Optimization

1. **Configure W&B**:

```bash
wandb login  # Enter your API key
```

2. **Run the optimization notebook**:

   - Open `Notebooks/pipelineTest3_optuna+w&b.ipynb`
   - Configure your W&B project name
   - Set the database URL (optional, defaults to local SQLite)
   - Execute all cells

3. **Key Features**:
   - Multi-objective optimization (MAE, memory, training time, throughput, latency)
   - Automatic model checkpointing to W&B
   - Real-time monitoring and logging
   - Pareto frontier analysis
   - Remote database persistence

### Optimization Objectives

The system optimizes for:

- **Validation MAE** (primary metric)
- **Model parameters** (complexity)
- **Training time** (speed)
- **Inference latency** (deployment readiness)

## 📓 Notebooks

The `Notebooks/` directory contains comprehensive analysis and experiments:

- **`1 - DatasetAnalysis.ipynb`**: Complete ZINC dataset exploration, statistics, and visualization
- **`2 - FeatureSelection.ipynb`**: Feature engineering approach using filter and wrapper methods for molecular property prediction
- **`pipelineTest3_optuna+w&b.ipynb`**: Main hyperparameter optimization with Optuna and W&B integration
- **`Study_Analysis.ipynb`**: **In-depth analysis of Optuna hyperparameter optimization results** with comprehensive performance evaluation and statistical testing
- **`KaggleNotebook.ipynb`**: Kaggle-optimized version for remote execution with GPU acceleration [DEPRECATED]
- **`LoopDetection.ipynb`**: Experimental graph topology analysis (loop detection)
- **`test_HGPSL.ipynb`**: Testing of Hierarchical Graph Pooling with Structure Learning (experimental)

## 🎯 Key Findings from Hyperparameter Optimization

Based on comprehensive analysis of 120+ Optuna trials (`Study_Analysis.ipynb`), here are the **statistically validated optimal configurations**:

### 🏆 Best Performing Components

#### **Pooling Strategy**

- **Winner**: **Attentional Pooling** (MAE: 0.1787 ± 0.0051)
- **Runner-up**: Set2Set Pooling (MAE: 0.1799 ± 0.0064)
- Statistical significance confirmed (Kruskal-Wallis test, p < 0.001)

#### **Regressor Architecture**

- **Winner**: **MLP Regressor** (MAE: 0.1790 ± 0.0061)
- **Runner-up**: Ensemble Regressor (MAE: 0.1810 ± 0.0059)
- Statistical significance confirmed (Kruskal-Wallis test, p < 0.001)

### 💡 Recommended Configuration

For optimal performance on ZINC dataset:

```python
# Optimal configuration based on 120+ trials
optimal_config = {
    'gnn_layer_name': 'GINEConv',          # Best overall GNN architecture
    'gnn_hidden_dim': 128-256,             # Sweet spot for complexity/performance
    'gnn_num_layers': 4-6,                 # Depth for molecular representations
    'pooling_type': 'attentional',         # Statistically best pooling
    'regressor_type': 'mlp',               # Statistically best regressor
    'learning_rate': 1e-3,                 # Stable convergence
    'batch_size': 128                      # Memory/performance balance
}
```

### 📊 Performance Insights

- **GNN Architecture Impact**: 40% of performance variance attributed to GNN configuration
- **Pooling Strategy Impact**: 25% of performance variance, with attentional pooling consistently outperforming alternatives
- **Regressor Type Impact**: 20% of performance variance, with MLP showing superior generalization
- **Best Achieved MAE**: **0.1696** (top 1% of trials)

> **Note**: All findings are based on rigorous statistical analysis with multiple validation approaches and significance testing. See `Study_Analysis.ipynb` for detailed methodology and visualizations.

## 📚 Documentation

For detailed technical documentation, API reference, and advanced usage:

- **[Package Documentation](src/ACAgraphML/README.md)**: Comprehensive technical documentation
- **[Examples](examples/)**: Complete usage examples and tutorials

## 🏗️ Development

### Project Structure

```
src/ACAgraphML/           # Main package
├── Pipeline/             # Core pipeline components
│   ├── Models/          # GNN, Pooling, Regressor, GDLPipeline
│   └── LightningModules/ # PyTorch Lightning integration
├── Dataset/             # Data loading and handling
├── Transforms/          # Data preprocessing and augmentation
├── Plotting/           # Visualization utilities
├── HGPSL/              # Experimental hierarchical pooling
└── utils.py            # Utility functions

tests/                   # Comprehensive test suite
examples/               # Usage examples
Notebooks/             # Analysis and experiments
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
