from torch_geometric.transforms import BaseTransform
import torch
from torch_geometric.data.data import Data
from torch.nn.functional import one_hot


class OneHotEdgeFeats(BaseTransform):
    """
    A PyTorch Geometric transform that applies one-hot encoding to the edge attributes.
    This transform is useful for converting categorical edge features into a
    one-hot encoded format, which is often required for machine learning models.
    The transform expects the edge attributes to be integer labels representing
    categories, and it will convert these labels into one-hot encoded vectors.

    Attributes:
        nClasses (int): The number of unique classes for one-hot encoding.
        categorical_dims (list, optional): List of dimensions to apply one-hot encoding to.
                                         If None, applies to all dimensions.
    """

    def __init__(self, nClasses: int, categorical_dims: list = None):
        """
        Initialize the OneHotEdgeFeats transform.

        Args:
            nClasses (int): The number of unique classes for one-hot encoding.
            categorical_dims (list, optional): List of dimensions to apply one-hot encoding to.
                                             If None, applies to all dimensions.
        """
        assert nClasses > 0, "Number of classes must be > 0."
        self.nClasses = nClasses
        self.categorical_dims = categorical_dims

    def __call__(self, data: Data):
        """
        Applies one-hot encoding to the edge attributes of the input data.

        Args:
            data (Data): A PyTorch Geometric Data object.

        Returns:
            Data: The transformed data with one-hot encoded edge attributes.
        """
        if data.edge_attr is None:
            raise ValueError(
                "Input data must have edge attributes (data.edge_attr).")

        edge_attr = data.edge_attr

        # Handle different input shapes
        if edge_attr.dim() == 1:
            # If edge_attr is 1D, reshape to 2D
            edge_attr = edge_attr.unsqueeze(-1)

        if self.categorical_dims is None:
            # Apply one-hot encoding to all dimensions
            if edge_attr.shape[1] == 1:
                # Single feature column
                if not torch.is_floating_point(edge_attr):
                    data.edge_attr = one_hot(
                        edge_attr.long(), num_classes=self.nClasses
                    ).squeeze(1).float()
                else:
                    # If already floating point, assume it's already processed
                    data.edge_attr = edge_attr.float()
            else:
                # Multiple feature columns - apply one-hot to each
                encoded_features = []
                for dim in range(edge_attr.shape[1]):
                    feature_col = edge_attr[:, dim]
                    if not torch.is_floating_point(feature_col):
                        encoded_col = one_hot(
                            feature_col.long(), num_classes=self.nClasses
                        ).float()
                        encoded_features.append(encoded_col)
                    else:
                        # Keep floating point features as is
                        encoded_features.append(feature_col.unsqueeze(-1))

                data.edge_attr = torch.cat(encoded_features, dim=-1)
        else:
            # Apply one-hot encoding only to specified dimensions
            encoded_features = []
            for dim in range(edge_attr.shape[1]):
                feature_col = edge_attr[:, dim]
                if dim in self.categorical_dims:
                    if not torch.is_floating_point(feature_col):
                        encoded_col = one_hot(
                            feature_col.long(), num_classes=self.nClasses
                        ).float()
                        encoded_features.append(encoded_col)
                    else:
                        # If already floating point, assume it's already processed
                        encoded_features.append(feature_col.unsqueeze(-1))
                else:
                    # Keep non-categorical dimensions as is
                    encoded_features.append(feature_col.unsqueeze(-1))

            data.edge_attr = torch.cat(encoded_features, dim=-1)

        return data

    def __repr__(self):
        """
        Returns a string representation of the transform.
        """
        return (f'{self.__class__.__name__}(nClasses={self.nClasses}, '
                f'categorical_dims={self.categorical_dims})')
