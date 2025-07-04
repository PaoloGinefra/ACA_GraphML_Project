from torch_geometric.data import Dataset, InMemoryDataset
import torch
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
sns.set_theme()


class TargetNormalizer:
    def __init__(self, verbose=False, outlier_threshold=3):
        """
        Initialize the TargetNormalizer.

        Args:
            verbose (bool): If True, print additional information during processing.
            outlier_threshold (float): The z-score threshold to consider a target as an outlier.
        """
        self.verbose = verbose
        self.target_mean = None
        self.target_std = None
        self.outlier_threshold = outlier_threshold

    def plotTargetDistribution(self, targets: torch.Tensor, title_suffix=""):
        """
        Plot the target distribution to visualize the data before and after normalization.

        Args:
            targets (torch.Tensor): The target values to plot.
            title_suffix (str): Additional text to add to the plot title.
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram
        ax[0].hist(targets.numpy(), bins=50, alpha=0.7, density=True)
        ax[0].set_title(f'Target Distribution{title_suffix}')
        ax[0].set_xlabel('Target Values')
        ax[0].set_ylabel('Density')
        mean = self.target_mean if self.target_mean is not None else targets.mean().item()
        std = self.target_std if self.target_std is not None else targets.std().item()
        ax[0].axvline(mean, color='red', linestyle='--',
                      label=f'Mean: {mean:.4f}')
        ax[0].axvline(mean + std, color='orange',
                      linestyle='--', alpha=0.7, label=f'±1 Std: {std:.4f}')
        ax[0].axvline(mean - std, color='orange',
                      linestyle='--', alpha=0.7)
        # Add vlines for k_outliers * std
        k_outliers = self.outlier_threshold if hasattr(
            self, 'outlier_threshold') else 3
        ax[0].axvline(mean + k_outliers * std, color='purple', linestyle=':', alpha=0.8,
                      label=f'+{k_outliers} Std (Outlier)')
        ax[0].axvline(mean - k_outliers * std, color='purple', linestyle=':', alpha=0.8,
                      label=f'-{k_outliers} Std (Outlier)')
        ax[0].legend()

        # Box plot
        ax[1].boxplot([targets.numpy()], labels=['Targets'])
        ax[1].set_title(f'Target Box Plot{title_suffix}')
        ax[1].set_ylabel('Target Values')

        plt.tight_layout()
        plt.show()

    def fit(self, dataset: Dataset):
        """
        Compute normalization statistics from the dataset.

        Args:
            dataset (Dataset): The PyTorch Geometric dataset to analyze.
        """
        # Extract all targets from the dataset
        all_targets = torch.cat([data.y for data in dataset])

        # Compute statistics
        self.target_mean = all_targets.mean().item()
        self.target_std = all_targets.std().item()

        if self.verbose:
            print(
                f"Target statistics: mean={self.target_mean:.4f}, std={self.target_std:.4f}")
            print(
                f"Range: [{all_targets.min().item():.4f}, {all_targets.max().item():.4f}]")

            self.plotTargetDistribution(all_targets, " (Original)")

        return self

    def normalize(self, dataset: Dataset) -> Dataset:
        """
        Apply normalization to all samples in the dataset.

        Args:
            dataset (Dataset): The PyTorch Geometric dataset to normalize.

        Returns:
            Dataset: A new dataset instance with normalized targets.
        """
        if self.target_mean is None or self.target_std is None:
            raise RuntimeError(
                "Call fit() first to compute normalization statistics.")

        normalized_ys = [
            (data.y - self.target_mean) / self.target_std if hasattr(data,
                                                                     'y') and data.y is not None else None
            for data in dataset
        ]
        normalized_ys = torch.stack(
            [n for n in normalized_ys if n is not None])

        inliersIndicies = torch.abs(normalized_ys) < self.outlier_threshold
        if inliersIndicies.any():
            dataset = dataset.copy(idx=inliersIndicies)

        # Create a copy of the dataset first
        try:
            # Standard PyTorch Geometric InMemoryDataset interface
            newDataset = dataset.__class__(
                root=dataset.root, transform=None, pre_transform=None)
            newDataset.data, newDataset.slices = dataset.data, dataset.slices
        except (TypeError, AttributeError):
            # Handle custom datasets that don't follow the standard interface
            try:
                # Check if the dataset has a simple constructor that takes a data_list
                newDataset = dataset.__class__(dataset.data_list)
            except TypeError:
                # Fallback: try to call constructor with no arguments and set data manually
                newDataset = dataset.__class__()
                newDataset.data_list = dataset.data_list

        self.target_mean = newDataset.data.y.mean().item()
        self.target_std = newDataset.data.y.std().item()

        newDataset._data.y = (
            newDataset.y - self.target_mean) / self.target_std

        if self.verbose:
            # Plot the distribution after normalization
            normalized_targets = torch.cat([data.y for data in newDataset])
            self.plotTargetDistribution(normalized_targets, " (Normalized)")

        return newDataset

    def __call__(self, data):
        """
        Transform individual data objects (for compatibility when used as a transform).

        Args:
            data: A PyTorch Geometric Data object.

        Returns:
            data: The transformed data object with normalized targets.
        """
        if self.target_mean is None or self.target_std is None:
            raise RuntimeError(
                "Call fit() first to compute normalization statistics.")

        # Normalize the target
        if hasattr(data, 'y') and data.y is not None:
            data.y = (data.y - self.target_mean) / self.target_std

        return data

    def denormalize(self, normalized_targets: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized targets back to original scale.

        Args:
            normalized_targets (torch.Tensor): Normalized target values.

        Returns:
            torch.Tensor: Denormalized target values in original scale.

        Raises:
            RuntimeError: If normalization statistics haven't been computed yet.
        """
        if self.target_mean is None or self.target_std is None:
            raise RuntimeError("Normalization statistics not available. "
                               "Call fit() first to compute statistics.")

        return normalized_targets * self.target_std + self.target_mean

    def get_statistics(self):
        """
        Get the computed normalization statistics.

        Returns:
            dict: Dictionary containing 'mean' and 'std' if available, None otherwise.
        """
        if self.target_mean is not None and self.target_std is not None:
            return {
                'mean': self.target_mean,
                'std': self.target_std
            }
        return None

    def __repr__(self):
        """
        Returns a string representation of the normalizer.
        """
        stats_str = ""
        if self.target_mean is not None and self.target_std is not None:
            stats_str = f", mean={self.target_mean:.4f}, std={self.target_std:.4f}"

        return (f'{self.__class__.__name__}(verbose={self.verbose}{stats_str})')
