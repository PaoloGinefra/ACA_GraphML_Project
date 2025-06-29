# ACA_GraphML_Project

Advanced Graph Machine Learning project for molecular property prediction using Graph Neural Networks (GNNs).

## Installation

### From GitHub (Recommended)

```bash
pip install git+https://github.com/PaoloGinefra/ACA_GraphML_Project.git
```

### From Local Clone

```bash
git clone https://github.com/PaoloGinefra/ACA_GraphML_Project.git
cd ACA_GraphML_Project
pip install -e .
```

### With Optional Dependencies

```bash
# For hyperparameter optimization
pip install "git+https://github.com/PaoloGinefra/ACA_GraphML_Project.git[optuna]"

# For Weights & Biases logging
pip install "git+https://github.com/PaoloGinefra/ACA_GraphML_Project.git[wandb]"

# For development
pip install "git+https://github.com/PaoloGinefra/ACA_GraphML_Project.git[dev]"

# All optional dependencies
pip install "git+https://github.com/PaoloGinefra/ACA_GraphML_Project.git[optuna,wandb,kaggle,chemistry,dev]"
```

## Quick Start

```python
from ACAgraphML import GDLPipeline, GDLPipelineLightningModule
from ACAgraphML.Pipeline.LightningModules.GDLPipelineLighningModule import create_lightning_standard

# Create a standard pipeline
model = create_lightning_standard(
    node_features=28,
    edge_features=4,
    lr=1e-3
)
```

## Features

- **GDL Pipeline**: Comprehensive graph neural network pipeline
- **Lightning Integration**: PyTorch Lightning wrapper for easy training
- **Multiple Architectures**: Support for various GNN architectures (GCN, GAT, GIN, etc.)
- **Hyperparameter Optimization**: Built-in Optuna support
- **Comprehensive Testing**: Extensive test suite
- **ZINC Dataset Support**: Ready-to-use molecular property prediction

## Documentation

See the individual module READMEs for detailed documentation:

- [GDL Pipeline Lightning Module](src/ACAgraphML/Pipeline/LightningModules/README.md)
- [GDL Pipeline Implementation](REGRESSOR_IMPLEMENTATION.md)

## License

MIT License - see [LICENSE](LICENSE) file for details.
