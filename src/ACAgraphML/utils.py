import matplotlib.pyplot as plt
import seaborn as sns
import torch
from typing import Dict, Any, Union
sns.set_theme()


def plotTrueVsPred(y, y_pred, y_val, y_val_pred, title='True vs Predicted Values', filename=None, hardLimit=True):
    """
    Plots true vs predicted values for both training and validation datasets.

    Parameters:
        y (array-like): True target values for the training set.
        y_pred (array-like): Predicted values for the training set.
        y_val (array-like): True target values for the validation set.
        y_val_pred (array-like): Predicted values for the validation set.
        title (str, optional): Title of the plot. Defaults to 'True vs Predicted Values'.
        filename (str, optional): If provided, saves the plot to the specified file path.
        hardLimit (bool, optional): If True, sets x and y limits to [-10, 5]. Defaults to True.

    The function creates a scatter plot comparing true and predicted values for both
    training and validation sets, includes a reference line (y = x), and optionally saves the plot.
    """
    plt.figure(figsize=(8, 5))
    plt.plot([y_val.min(), y_val.max()], [
             y_val.min(), y_val.max()], 'r--', lw=2, alpha=0.5)
    plt.scatter(y, y_pred, alpha=0.5, s=2,
                label='Train Predictions', color='orange')
    plt.scatter(y_val, y_val_pred, alpha=0.5, s=2,
                label='Validation Predictions', color='blue')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(title)
    if hardLimit:
        plt.xlim(-10, 5)
        plt.ylim(-10, 5)
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def get_model_memory_usage(model: torch.nn.Module) -> Dict[str, float]:
    """
    Calculate memory consumption of a PyTorch model.

    Args:
        model: PyTorch model to analyze

    Returns:
        dict: Memory usage information containing:
            - parameter_memory_mb: Memory used by model parameters (MB)
            - buffer_memory_mb: Memory used by model buffers (MB) 
            - total_memory_mb: Total model memory usage (MB)
            - parameter_memory_bytes: Memory used by parameters (bytes)
            - buffer_memory_bytes: Memory used by buffers (bytes)
            - total_memory_bytes: Total memory usage (bytes)
    """
    # Calculate parameter memory
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    # Convert to MB
    param_memory_mb = param_size / (1024 * 1024)
    buffer_memory_mb = buffer_size / (1024 * 1024)
    total_memory_mb = param_memory_mb + buffer_memory_mb

    return {
        'parameter_memory_mb': param_memory_mb,
        'buffer_memory_mb': buffer_memory_mb,
        'total_memory_mb': total_memory_mb,
        'parameter_memory_bytes': param_size,
        'buffer_memory_bytes': buffer_size,
        'total_memory_bytes': param_size + buffer_size
    }


def get_comprehensive_model_info(model: Union[torch.nn.Module, Any],
                                 include_hparams: bool = True) -> Dict[str, Any]:
    """
    Get comprehensive model information including parameter counts, memory usage, and hyperparameters.

    Args:
        model: Model to analyze (should have get_num_parameters() method and optionally hparams attribute)
        include_hparams: Whether to include hyperparameters from model.hparams if available

    Returns:
        dict: Comprehensive model information containing:
            - Parameter counts (total, gnn, pooling, regressor, other)
            - Memory usage (MB and bytes)
            - Parameter distribution percentages
            - Additional metrics (memory per parameter, parameters in millions)
            - Hyperparameters (if include_hparams=True and model has hparams)
    """
    # Get parameter counts
    if hasattr(model, 'get_num_parameters'):
        model_params = model.get_num_parameters()
    else:
        # Fallback for models without get_num_parameters method
        total_params = sum(p.numel() for p in model.parameters())
        model_params = {'total': total_params, 'gnn': 0,
                        'pooling': 0, 'regressor': 0, 'other': total_params}

    # Get memory usage
    model_memory = get_model_memory_usage(model)

    # Create comprehensive information dictionary
    model_info = {
        # Parameter counts
        'model/total_parameters': model_params['total'],
        'model/gnn_parameters': model_params['gnn'],
        'model/pooling_parameters': model_params['pooling'],
        'model/regressor_parameters': model_params['regressor'],
        'model/other_parameters': model_params['other'],

        # Memory usage
        'model/parameter_memory_mb': model_memory['parameter_memory_mb'],
        'model/buffer_memory_mb': model_memory['buffer_memory_mb'],
        'model/total_memory_mb': model_memory['total_memory_mb'],

        # Parameter distribution (percentages)
        'model/gnn_param_percentage': (model_params['gnn'] / model_params['total']) * 100 if model_params['total'] > 0 else 0,
        'model/pooling_param_percentage': (model_params['pooling'] / model_params['total']) * 100 if model_params['total'] > 0 else 0,
        'model/regressor_param_percentage': (model_params['regressor'] / model_params['total']) * 100 if model_params['total'] > 0 else 0,
        'model/other_param_percentage': (model_params['other'] / model_params['total']) * 100 if model_params['total'] > 0 else 0,

        # Additional analysis
        'model/memory_per_parameter_bytes': model_memory['total_memory_bytes'] / model_params['total'] if model_params['total'] > 0 else 0,
        'model/parameters_millions': model_params['total'] / 1_000_000,
    }

    # Include hyperparameters if available and requested
    if include_hparams and hasattr(model, 'hparams'):
        model_info.update(model.hparams)

    return model_info


def print_model_summary(model: Union[torch.nn.Module, Any]) -> None:
    """
    Print a formatted summary of model parameter counts and memory usage.

    Args:
        model: Model to analyze
    """
    # Get parameter counts
    if hasattr(model, 'get_num_parameters'):
        model_params = model.get_num_parameters()
    else:
        total_params = sum(p.numel() for p in model.parameters())
        model_params = {'total': total_params, 'gnn': 0,
                        'pooling': 0, 'regressor': 0, 'other': total_params}

    # Get memory usage
    model_memory = get_model_memory_usage(model)

    print("=== Model Parameter Breakdown ===")
    print(f"Total parameters: {model_params['total']:,}")
    print(f"GNN parameters: {model_params['gnn']:,}")
    print(f"Pooling parameters: {model_params['pooling']:,}")
    print(f"Regressor parameters: {model_params['regressor']:,}")
    print(f"Other parameters: {model_params['other']:,}")

    print(f"\n=== Model Memory Usage ===")
    print(f"Parameter memory: {model_memory['parameter_memory_mb']:.2f} MB")
    print(f"Buffer memory: {model_memory['buffer_memory_mb']:.2f} MB")
    print(f"Total memory: {model_memory['total_memory_mb']:.2f} MB")

    print(f"\n=== Parameter Distribution ===")
    if model_params['total'] > 0:
        print(
            f"GNN: {(model_params['gnn'] / model_params['total']) * 100:.1f}%")
        print(
            f"Pooling: {(model_params['pooling'] / model_params['total']) * 100:.1f}%")
        print(
            f"Regressor: {(model_params['regressor'] / model_params['total']) * 100:.1f}%")
        print(
            f"Other: {(model_params['other'] / model_params['total']) * 100:.1f}%")
