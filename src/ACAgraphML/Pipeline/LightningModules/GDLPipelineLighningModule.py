"""
PyTorch Lightning Module for GDLPipeline.

This module provides a comprehensive PyTorch Lightning wrapper for the GDLPipeline
with extensive hyperparameter optimization support, advanced optimization strategies,
and comprehensive logging capabilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Union, Literal, List, Tuple
import warnings
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
    ExponentialLR,
    CosineAnnealingWarmRestarts
)

from ..Models.GDLPipeline import (
    GDLPipeline,
    GNNConfig,
    PoolingConfig,
    RegressorConfig,
    create_baseline_pipeline,
    create_standard_pipeline,
    create_advanced_pipeline,
    create_lightweight_pipeline,
    create_attention_pipeline
)


class GDLPipelineLightningModule(pl.LightningModule):
    """
    PyTorch Lightning Module for GDLPipeline.

    This module wraps the GDLPipeline with comprehensive training, validation,
    and testing capabilities, designed for extensive hyperparameter optimization.

    Features:
    - Multiple loss functions and metrics
    - Advanced optimization strategies with various schedulers
    - Comprehensive logging and monitoring
    - Gradient clipping and regularization
    - Early stopping and model checkpointing support
    - Target normalization and denormalization
    - Embedding extraction capabilities
    - Support for all GDLPipeline configurations
    """

    def __init__(
        self,
        # Model configuration
        node_features: int,
        edge_features: Optional[int] = None,
        pipeline_config: Literal["baseline", "standard", "advanced",
                                 "lightweight", "attention", "custom"] = "standard",

        # Custom pipeline configuration (used if pipeline_config="custom")
        gnn_config: Optional[Union[GNNConfig, Dict[str, Any]]] = None,
        pooling_config: Optional[Union[PoolingConfig, Dict[str, Any]]] = None,
        regressor_config: Optional[Union[RegressorConfig,
                                         Dict[str, Any]]] = None,
        global_dropout: float = 0.0,
        use_batch_norm: bool = False,

        # Loss and metrics
        loss: Literal['mse', 'mae', 'huber', 'smooth_l1'] = 'mae',
        huber_delta: float = 1.0,

        # Optimization
        optimizer: Literal['adam', 'adamw', 'sgd', 'rmsprop'] = 'adamw',
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        momentum: float = 0.9,  # For SGD

        # Learning rate scheduling
        lr_scheduler: Optional[Literal['cosine', 'plateau', 'step',
                                       'exponential', 'cosine_restarts', 'none']] = 'cosine',
        lr_scheduler_params: Optional[Dict[str, Any]] = None,
        warmup_epochs: int = 0,

        # Regularization
        gradient_clip_val: Optional[float] = 1.0,
        gradient_clip_algorithm: Literal['norm', 'value'] = 'norm',

        # Target normalization
        target_mean: Optional[float] = None,
        target_std: Optional[float] = None,

        # Monitoring and logging
        monitor_metric: str = 'val_mae',
        log_embeddings: bool = False,
        log_predictions: bool = False,

        # Advanced features
        label_smoothing: float = 0.0,

        # Pipeline specific parameters
        **pipeline_kwargs
    ):
        """
        Initialize the GDLPipeline Lightning Module.

        Args:
            node_features: Number of node features
            edge_features: Number of edge features (None if not using edges)
            pipeline_config: Pre-defined pipeline configuration
            gnn_config: Custom GNN configuration (for pipeline_config="custom")
            pooling_config: Custom pooling configuration (for pipeline_config="custom")
            regressor_config: Custom regressor configuration (for pipeline_config="custom")
            global_dropout: Global dropout rate
            use_batch_norm: Whether to use batch normalization
            loss: Loss function type
            huber_delta: Delta parameter for Huber loss
            optimizer: Optimizer type
            lr: Learning rate
            weight_decay: Weight decay for regularization
            betas: Beta parameters for Adam/AdamW
            momentum: Momentum for SGD
            lr_scheduler: Learning rate scheduler type
            lr_scheduler_params: Additional scheduler parameters
            warmup_epochs: Number of warmup epochs
            gradient_clip_val: Gradient clipping value
            gradient_clip_algorithm: Gradient clipping algorithm
            target_mean: Target mean for normalization
            target_std: Target std for normalization
            monitor_metric: Metric to monitor for callbacks
            log_embeddings: Whether to log embeddings
            log_predictions: Whether to log predictions
            label_smoothing: Label smoothing factor
            **pipeline_kwargs: Additional pipeline arguments
        """
        # Save all hyperparameters
        self.save_hyperparameters()

        super().__init__()

        # Store configuration
        self.node_features = node_features
        self.edge_features = edge_features
        self.monitor_metric = monitor_metric
        self.log_embeddings = log_embeddings
        self.log_predictions = log_predictions

        # Create pipeline based on configuration
        self.pipeline = self._create_pipeline(
            pipeline_config=pipeline_config,
            gnn_config=gnn_config,
            pooling_config=pooling_config,
            regressor_config=regressor_config,
            global_dropout=global_dropout,
            use_batch_norm=use_batch_norm,
            target_mean=target_mean,
            target_std=target_std,
            **pipeline_kwargs
        )

        # Set up loss function
        self.loss_fn = self._create_loss_function(
            loss, huber_delta, label_smoothing)

        # Store optimization parameters
        self.optimizer_name = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.momentum = momentum
        self.lr_scheduler_name = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params or {}
        self.warmup_epochs = warmup_epochs

        # Store regularization parameters
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm

        # Initialize validation outputs storage for epoch-end logging
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def _create_pipeline(
        self,
        pipeline_config: str,
        gnn_config: Optional[Union[GNNConfig, Dict[str, Any]]],
        pooling_config: Optional[Union[PoolingConfig, Dict[str, Any]]],
        regressor_config: Optional[Union[RegressorConfig, Dict[str, Any]]],
        global_dropout: float,
        use_batch_norm: bool,
        target_mean: Optional[float],
        target_std: Optional[float],
        **pipeline_kwargs
    ) -> GDLPipeline:
        """Create the GDLPipeline based on configuration."""

        if pipeline_config == "baseline":
            pipeline = create_baseline_pipeline(
                self.node_features, self.edge_features)
        elif pipeline_config == "standard":
            pipeline = create_standard_pipeline(
                self.node_features, self.edge_features)
        elif pipeline_config == "advanced":
            pipeline = create_advanced_pipeline(
                self.node_features, self.edge_features)
        elif pipeline_config == "lightweight":
            pipeline = create_lightweight_pipeline(
                self.node_features, self.edge_features)
        elif pipeline_config == "attention":
            pipeline = create_attention_pipeline(
                self.node_features, self.edge_features)
        elif pipeline_config == "custom":
            # Create custom pipeline
            pipeline = GDLPipeline(
                node_features=self.node_features,
                edge_features=self.edge_features,
                gnn_config=gnn_config,
                pooling_config=pooling_config,
                regressor_config=regressor_config,
                global_dropout=global_dropout,
                use_batch_norm=use_batch_norm,
                target_mean=target_mean,
                target_std=target_std,
                **pipeline_kwargs
            )
        else:
            raise ValueError(f"Unknown pipeline_config: {pipeline_config}")

        # Set target normalization if provided
        if target_mean is not None and target_std is not None:
            pipeline.set_target_normalization(target_mean, target_std)

        return pipeline

    def _create_loss_function(self, loss: str, huber_delta: float, label_smoothing: float) -> nn.Module:
        """Create the loss function."""
        if loss == 'mse':
            if label_smoothing > 0:
                warnings.warn(
                    "Label smoothing not supported for MSE loss, ignoring.")
            return nn.MSELoss()
        elif loss == 'mae':
            if label_smoothing > 0:
                warnings.warn(
                    "Label smoothing not supported for MAE loss, ignoring.")
            return nn.L1Loss()
        elif loss == 'huber':
            if label_smoothing > 0:
                warnings.warn(
                    "Label smoothing not supported for Huber loss, ignoring.")
            return nn.HuberLoss(delta=huber_delta)
        elif loss == 'smooth_l1':
            if label_smoothing > 0:
                warnings.warn(
                    "Label smoothing not supported for SmoothL1 loss, ignoring.")
            return nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss function: {loss}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass through the pipeline."""
        return self.pipeline(x, edge_index, edge_attr, batch, return_embeddings)

    def _shared_step(self, batch, batch_idx: int, stage: str) -> Dict[str, torch.Tensor]:
        """Shared step for training, validation, and testing."""
        # Extract data from batch
        x, edge_index, batch_indices = batch.x, batch.edge_index, batch.batch
        edge_attr = getattr(batch, 'edge_attr', None)
        targets = batch.y

        # Ensure correct data types
        x = x.float()
        edge_index = edge_index.long()
        batch_indices = batch_indices.long()
        if edge_attr is not None:
            edge_attr = edge_attr.float()
        targets = targets.float()

        # Forward pass
        if self.log_embeddings and stage == 'val':
            predictions, embeddings = self.pipeline(
                x, edge_index, edge_attr, batch_indices, return_embeddings=True
            )
        else:
            predictions = self.pipeline(
                x, edge_index, edge_attr, batch_indices)
            embeddings = None

        # Squeeze predictions if needed
        predictions = predictions.squeeze(-1)
        if targets.dim() > 1:
            targets = targets.squeeze(-1)

        # Compute loss
        loss = self.loss_fn(predictions, targets)

        # Compute metrics
        mae = nn.L1Loss()(predictions, targets)
        mse = nn.MSELoss()(predictions, targets)
        rmse = torch.sqrt(mse)

        # Compute additional metrics
        with torch.no_grad():
            abs_error = torch.abs(predictions - targets)
            mean_abs_error = torch.mean(abs_error)
            std_abs_error = torch.std(abs_error)
            max_abs_error = torch.max(abs_error)

            # R² score
            ss_res = torch.sum((targets - predictions) ** 2)
            ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
            r2_score = 1 - (ss_res / (ss_tot + 1e-8))

        return {
            'loss': loss,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mean_abs_error': mean_abs_error,
            'std_abs_error': std_abs_error,
            'max_abs_error': max_abs_error,
            'r2_score': r2_score,
            'predictions': predictions.detach(),
            'targets': targets.detach(),
            'embeddings': embeddings
        }

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Training step."""
        outputs = self._shared_step(batch, batch_idx, 'train')

        # Determine batch size for logging
        batch_size = len(torch.unique(batch.batch)) if hasattr(
            batch, 'batch') else len(batch.y)

        # Log training metrics
        self.log('train_loss', outputs['loss'],
                 prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_mae', outputs['mae'],
                 prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('train_rmse', outputs['rmse'], on_step=False,
                 on_epoch=True, batch_size=batch_size)
        self.log('train_r2', outputs['r2_score'],
                 on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('train_max_error',
                 outputs['max_abs_error'], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('train_mean_abs_error',
                 outputs['mean_abs_error'], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('train_std_abs_error',
                 outputs['std_abs_error'], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('train_predictions_mean',
                 outputs['predictions'].mean(), on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('train_predictions_std',
                 outputs['predictions'].std(), on_step=False, on_epoch=True, batch_size=batch_size)

        # Log learning rate (only when properly attached to trainer)
        try:
            if self.lr_schedulers() and hasattr(self, '_trainer') and self._trainer is not None:
                current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
                self.log('lr', current_lr, prog_bar=True,
                         on_step=True, on_epoch=False, batch_size=batch_size)
        except (RuntimeError, AttributeError):
            # Skip LR logging if not attached to trainer
            pass

        return outputs['loss']

    def validation_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        outputs = self._shared_step(batch, batch_idx, 'val')

        # Determine batch size for logging
        batch_size = len(torch.unique(batch.batch)) if hasattr(
            batch, 'batch') else len(batch.y)

        # Log validation metrics
        self.log('val_loss', outputs['loss'],
                 prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_mae', outputs['mae'],
                 prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_rmse', outputs['rmse'], on_step=False,
                 on_epoch=True, batch_size=batch_size)
        self.log('val_r2', outputs['r2_score'], on_step=False,
                 on_epoch=True, batch_size=batch_size)
        self.log('val_max_error',
                 outputs['max_abs_error'], on_step=False, on_epoch=True, batch_size=batch_size)

        # Store outputs for epoch-end processing
        self.validation_step_outputs.append(outputs)

        return outputs

    def test_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        outputs = self._shared_step(batch, batch_idx, 'test')

        # Determine batch size for logging
        batch_size = len(torch.unique(batch.batch)) if hasattr(
            batch, 'batch') else len(batch.y)

        # Log test metrics
        self.log('test_loss', outputs['loss'],
                 prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_mae', outputs['mae'],
                 prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_rmse', outputs['rmse'], on_step=False,
                 on_epoch=True, batch_size=batch_size)
        self.log('test_r2', outputs['r2_score'],
                 on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_max_error',
                 outputs['max_abs_error'], on_step=False, on_epoch=True, batch_size=batch_size)

        # Store outputs for epoch-end processing
        self.test_step_outputs.append(outputs)

        return outputs

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        if not self.validation_step_outputs:
            return

        # Aggregate predictions and targets
        predictions_list = []
        targets_list = []

        for x in self.validation_step_outputs:
            pred = x['predictions']
            target = x['targets']

            # Ensure tensors have proper dimensions for concatenation
            if pred.numel() > 0:
                # If 0-dimensional, add a dimension
                if pred.dim() == 0:
                    pred = pred.unsqueeze(0)
                predictions_list.append(pred)

            if target.numel() > 0:
                # If 0-dimensional, add a dimension
                if target.dim() == 0:
                    target = target.unsqueeze(0)
                targets_list.append(target)

        if not predictions_list or not targets_list:
            return

        all_predictions = torch.cat(predictions_list)
        all_targets = torch.cat(targets_list)

        # Log prediction statistics
        pred_mean = torch.mean(all_predictions)
        pred_std = torch.std(all_predictions)
        target_mean = torch.mean(all_targets)
        target_std = torch.std(all_targets)

        total_samples = len(all_predictions)

        self.log('val_pred_mean', pred_mean,
                 on_epoch=True, batch_size=total_samples)
        self.log('val_pred_std', pred_std,
                 on_epoch=True, batch_size=total_samples)

        # Log prediction vs target correlation
        if len(all_predictions) > 1:
            correlation = torch.corrcoef(torch.stack(
                [all_predictions, all_targets]))[0, 1]
            self.log('val_correlation', correlation,
                     on_epoch=True, batch_size=total_samples)

        # Clear outputs
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        """Called at the end of test epoch."""
        if not self.test_step_outputs:
            return

        # Aggregate predictions and targets
        predictions_list = []
        targets_list = []

        for x in self.test_step_outputs:
            pred = x['predictions']
            target = x['targets']

            # Ensure tensors have proper dimensions for concatenation
            if pred.numel() > 0:
                # If 0-dimensional, add a dimension
                if pred.dim() == 0:
                    pred = pred.unsqueeze(0)
                predictions_list.append(pred)

            if target.numel() > 0:
                # If 0-dimensional, add a dimension
                if target.dim() == 0:
                    target = target.unsqueeze(0)
                targets_list.append(target)

        if not predictions_list or not targets_list:
            return

        all_predictions = torch.cat(predictions_list)
        all_targets = torch.cat(targets_list)

        # Log prediction statistics
        pred_mean = torch.mean(all_predictions)
        pred_std = torch.std(all_predictions)
        target_mean = torch.mean(all_targets)
        target_std = torch.std(all_targets)

        total_samples = len(all_predictions)

        self.log('test_pred_mean', pred_mean,
                 on_epoch=True, batch_size=total_samples)
        self.log('test_pred_std', pred_std,
                 on_epoch=True, batch_size=total_samples)
        self.log('test_target_mean', target_mean,
                 on_epoch=True, batch_size=total_samples)
        self.log('test_target_std', target_std,
                 on_epoch=True, batch_size=total_samples)

        # Log prediction vs target correlation
        if len(all_predictions) > 1:
            correlation = torch.corrcoef(torch.stack(
                [all_predictions, all_targets]))[0, 1]
            self.log('test_correlation', correlation,
                     on_epoch=True, batch_size=total_samples)

        # Clear outputs
        self.test_step_outputs.clear()

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        """Configure optimizers and learning rate schedulers."""
        # Create optimizer
        if self.optimizer_name == 'adam':
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=self.betas
            )
        elif self.optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=self.betas
            )
        elif self.optimizer_name == 'sgd':
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum
            )
        elif self.optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        # Add warmup if requested
        if self.warmup_epochs > 0:
            def warmup_lambda(epoch):
                if epoch < self.warmup_epochs:
                    return (epoch + 1) / self.warmup_epochs
                return 1.0

            warmup_scheduler = optim.lr_scheduler.LambdaLR(
                optimizer, warmup_lambda)

            if self.lr_scheduler_name is None or self.lr_scheduler_name == 'none':
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': warmup_scheduler,
                        'interval': 'epoch'
                    }
                }

        # Create learning rate scheduler if requested
        if self.lr_scheduler_name is None or self.lr_scheduler_name == 'none':
            return optimizer

        scheduler_config = {'interval': 'epoch'}

        if self.lr_scheduler_name == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.lr_scheduler_params.get('T_max', 100),
                eta_min=self.lr_scheduler_params.get('eta_min', self.lr * 0.01)
            )
        elif self.lr_scheduler_name == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.lr_scheduler_params.get('factor', 0.5),
                patience=self.lr_scheduler_params.get('patience', 10)
            )
            scheduler_config['monitor'] = self.monitor_metric
            scheduler_config['reduce_on_plateau'] = True
        elif self.lr_scheduler_name == 'step':
            scheduler = StepLR(
                optimizer,
                step_size=self.lr_scheduler_params.get('step_size', 30),
                gamma=self.lr_scheduler_params.get('gamma', 0.1)
            )
        elif self.lr_scheduler_name == 'exponential':
            scheduler = ExponentialLR(
                optimizer,
                gamma=self.lr_scheduler_params.get('gamma', 0.95)
            )
        elif self.lr_scheduler_name == 'cosine_restarts':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.lr_scheduler_params.get('T_0', 10),
                T_mult=self.lr_scheduler_params.get('T_mult', 2),
                eta_min=self.lr_scheduler_params.get('eta_min', self.lr * 0.01)
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.lr_scheduler_name}")

        # Combine with warmup if needed
        if self.warmup_epochs > 0:
            try:
                from torch.optim.lr_scheduler import SequentialLR

                def warmup_lambda(epoch):
                    return (epoch + 1) / self.warmup_epochs

                warmup_scheduler = optim.lr_scheduler.LambdaLR(
                    optimizer, warmup_lambda)
                combined_scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, scheduler],
                    milestones=[self.warmup_epochs]
                )
                scheduler_config['scheduler'] = combined_scheduler
            except ImportError:
                warnings.warn(
                    "SequentialLR not available in this PyTorch version. Using scheduler without warmup.")
                scheduler_config['scheduler'] = scheduler
        else:
            scheduler_config['scheduler'] = scheduler

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler_config
        }

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        """Configure gradient clipping."""
        if self.gradient_clip_val is not None:
            if self.gradient_clip_algorithm == 'norm':
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    max_norm=self.gradient_clip_val
                )
            elif self.gradient_clip_algorithm == 'value':
                torch.nn.utils.clip_grad_value_(
                    self.parameters(),
                    clip_value=self.gradient_clip_val
                )

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        embedding_type: Literal['node', 'graph', 'all'] = 'graph'
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Extract embeddings from the pipeline."""
        self.eval()
        with torch.no_grad():
            if embedding_type == 'node':
                return self.pipeline.get_node_embeddings(x, edge_index, edge_attr)
            elif embedding_type == 'graph':
                return self.pipeline.get_graph_embeddings(x, edge_index, edge_attr, batch)
            elif embedding_type == 'all':
                return self.pipeline.get_all_embeddings(x, edge_index, edge_attr, batch)
            else:
                raise ValueError(f"Unknown embedding_type: {embedding_type}")

    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Make predictions using the pipeline."""
        self.eval()
        with torch.no_grad():
            return self.pipeline.predict(x, edge_index, edge_attr, batch)

    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get the configuration of the underlying pipeline."""
        return self.pipeline.get_config()

    def get_num_parameters(self) -> Dict[str, int]:
        """Get the number of parameters in the pipeline."""
        return self.pipeline.get_num_parameters()


# Convenience functions for creating Lightning modules
def create_lightning_baseline(node_features: int, edge_features: Optional[int] = None, **kwargs) -> GDLPipelineLightningModule:
    """Create a baseline Lightning module."""
    return GDLPipelineLightningModule(
        node_features=node_features,
        edge_features=edge_features,
        pipeline_config="baseline",
        **kwargs
    )


def create_lightning_standard(node_features: int, edge_features: Optional[int] = None, **kwargs) -> GDLPipelineLightningModule:
    """Create a standard Lightning module."""
    return GDLPipelineLightningModule(
        node_features=node_features,
        edge_features=edge_features,
        pipeline_config="standard",
        **kwargs
    )


def create_lightning_advanced(node_features: int, edge_features: Optional[int] = None, **kwargs) -> GDLPipelineLightningModule:
    """Create an advanced Lightning module."""
    return GDLPipelineLightningModule(
        node_features=node_features,
        edge_features=edge_features,
        pipeline_config="advanced",
        **kwargs
    )


def create_lightning_lightweight(node_features: int, edge_features: Optional[int] = None, **kwargs) -> GDLPipelineLightningModule:
    """Create a lightweight Lightning module."""
    return GDLPipelineLightningModule(
        node_features=node_features,
        edge_features=edge_features,
        pipeline_config="lightweight",
        **kwargs
    )


def create_lightning_attention(node_features: int, edge_features: Optional[int] = None, **kwargs) -> GDLPipelineLightningModule:
    """Create an attention-focused Lightning module."""
    return GDLPipelineLightningModule(
        node_features=node_features,
        edge_features=edge_features,
        pipeline_config="attention",
        **kwargs
    )


def create_lightning_custom(
    node_features: int,
    edge_features: Optional[int] = None,
    gnn_config: Optional[Union[GNNConfig, Dict[str, Any]]] = None,
    pooling_config: Optional[Union[PoolingConfig, Dict[str, Any]]] = None,
    regressor_config: Optional[Union[RegressorConfig, Dict[str, Any]]] = None,
    **kwargs
) -> GDLPipelineLightningModule:
    """Create a custom Lightning module."""
    return GDLPipelineLightningModule(
        node_features=node_features,
        edge_features=edge_features,
        pipeline_config="custom",
        gnn_config=gnn_config,
        pooling_config=pooling_config,
        regressor_config=regressor_config,
        **kwargs
    )
