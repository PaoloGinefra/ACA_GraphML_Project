import torch


class PCAnodeFeatsTransform:
    """
    A class to transform node features using PCA.
    """

    def __init__(self, explained_variance_ratio=0.95):
        """
        Initialize the PCA transform with the specified number of components.

        Args:
            num_components (int): The number of principal components to keep.
        """
        self.explained_variance_ratio = explained_variance_ratio

    def __call__(self, data):
        """
        Transform the node features using the fitted PCA model.

        Args:
            data: The dataset containing node features.

        Returns:
            Transformed dataset with reduced node features.
        """
        _, S, V = torch.svd(data.x)
        total_variance = torch.sum(S ** 2)
        cumulative_variance = torch.cumsum(S ** 2, dim=0) / total_variance
        num_components = torch.sum(
            cumulative_variance < self.explained_variance_ratio).item() + 1

        principal_components = data.x @ V[:, :num_components]
        data.x = principal_components
        return data
