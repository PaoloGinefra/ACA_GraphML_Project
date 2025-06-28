"""
ACAgraphML - Advanced Graph Machine Learning for ACA project.

A comprehensive package for graph neural networks with focus on molecular property prediction.
"""

__version__ = "0.0.1"
__author__ = "Paolo Ginefra"
__email__ = "paolo.ginefra@gmail.com"

# Import main components
try:
    from .Pipeline.Models.GDLPipeline import GDLPipeline, GNNConfig, PoolingConfig, RegressorConfig
    from .Pipeline.LightningModules.GDLPipelineLighningModule import GDLPipelineLightningModule
except ImportError:
    # Handle optional imports gracefully
    pass

__all__ = [
    "GDLPipeline",
    "GNNConfig", 
    "PoolingConfig",
    "RegressorConfig",
    "GDLPipelineLightningModule"
]