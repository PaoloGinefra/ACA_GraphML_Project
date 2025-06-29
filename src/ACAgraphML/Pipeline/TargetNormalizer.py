from torch_geometric.data import Dataset, InMemoryDataset
import torch
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
sns.set_theme()


class TargetNormalizer:
    def __init__(self, verbose=False):
        """
        Initialize the TargetNormalizer.

        Args:
            verbose (bool): If True, print additional information during processing.
        """
        self.verbose = verbose
        self.target_mean = None
        self.target_std = None

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
        ax[0].axvline(targets.mean().item(), color='red', linestyle='--',
                      label=f'Mean: {targets.mean().item():.4f}')
        ax[0].axvline(targets.mean().item() + targets.std().item(), color='orange',
                      linestyle='--', alpha=0.7, label=f'±1 Std: {targets.std().item():.4f}')
        ax[0].axvline(targets.mean().item() - targets.std().item(), color='orange',
                      linestyle='--', alpha=0.7)
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

        # Now modify targets in place on the copied dataset
        for data in tqdm(newDataset, desc="Normalizing targets", leave=False):
            # Normalize the target in place
            if hasattr(data, 'y') and data.y is not None:
                data.y = (data.y - self.target_mean) / self.target_std

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
