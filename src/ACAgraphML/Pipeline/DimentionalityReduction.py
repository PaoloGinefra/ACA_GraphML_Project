from torch_geometric.data import Dataset
import torch
import copy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


class DimentionalityReduction:
    def __init__(self, explained_variance_ratio=0.95, verbose=False):
        """
        Initialize the DimensionalityReduction class with the desired explained variance ratio.

        Args:
            explained_variance_ratio (float): The desired explained variance ratio for PCA.
            verbose (bool): If True, print additional information during processing.
        """
        self.explained_variance_ratio = explained_variance_ratio
        self.verbose = verbose
        self.principal_directions = None
        self.num_components = None

    def plotSingularValues(self, singular_values: torch.Tensor):
        """
        Plot the singular values to visualize the explained variance.

        Args:
            singular_values (torch.Tensor): The singular values obtained from SVD.
        """
        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        ax[0].semilogy(singular_values, marker='o')
        ax[0].set_title('Singular Values')
        ax[0].set_xlabel('Index')
        ax[0].set_ylabel('Value')

        # Calculate cumulative explained variance
        total_variance = torch.sum(singular_values ** 2)
        cumulative_variance = torch.cumsum(
            singular_values ** 2, dim=0) / total_variance
        ax[1].plot(cumulative_variance, marker='o')
        ax[1].set_title('Cumulative Explained Variance')
        ax[1].set_xlabel('Number of Components')
        ax[1].set_ylabel('Cumulative Variance')
        ax[1].axhline(self.explained_variance_ratio, color='red', linestyle='--',
                      label=f'Explained Variance Ratio = {self.explained_variance_ratio}')
        ax[1].legend()

    def __call__(self, dataset: Dataset, useState=False) -> Dataset:
        """
        Apply PCA to reduce the dimensionality of node features in the dataset.

        Args:
            dataset (Dataset): The PyTorch Geometric dataset containing graph data.
            useState (bool): If True, use saved principal directions and number of components.

        Returns:
            Dataset: A new dataset with reduced node features.
        """
        normalized_features = dataset.data.x - \
            dataset.data.x.mean(dim=0, keepdim=True)

        if useState and self.principal_directions is not None and self.num_components is not None:
            V = self.principal_directions
            num_components = self.num_components
            if self.verbose:
                print("Using saved principal directions.")
        else:
            # Perform SVD on the node features
            _, S, V = torch.svd(normalized_features)

            if self.verbose:
                self.plotSingularValues(S)

            # Calculate total variance and cumulative variance
            total_variance = torch.sum(S ** 2)
            cumulative_variance = torch.cumsum(S ** 2, dim=0) / total_variance

            # Determine the number of components to keep
            num_components = torch.sum(
                cumulative_variance < self.explained_variance_ratio).item() + 1

            if self.verbose:
                print(f"Number of components to keep: {num_components}")
                print(
                    f"Explained variance ratio: {cumulative_variance[num_components - 1].item()}")

            # Save principal directions and number of components
            self.principal_directions = V
            self.num_components = num_components

        # Project node features onto the principal components
        projected = normalized_features @ V.T[:, :num_components]

        # Create a copy of the dataset to preserve all attributes
        newDataset = copy.deepcopy(dataset)

        # Replace only the node features with the projected ones
        newDataset.data.x = projected

        # If the dataset has a _data_list attribute, update it as well
        if hasattr(newDataset, '_data_list') and newDataset._data_list is not None:
            # Clear the data list to force regeneration with new features
            newDataset._data_list = [None] * len(newDataset)

        return newDataset
