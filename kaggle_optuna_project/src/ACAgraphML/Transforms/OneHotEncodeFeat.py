from torch_geometric.transforms import BaseTransform, OneHotDegree
from torch_geometric.utils import to_dense_adj
import torch
from torch_geometric.data.data import Data
from torch.nn.functional import one_hot


class OneHotEncodeFeat(BaseTransform):
    """
    A PyTorch Geometric transform that applies one-hot encoding to the node features.
    This transform is useful for converting categorical node features into a
    one-hot encoded format, which is often required for machine learning models.
    The transform expects the node features to be integer labels representing
    categories, and it will convert these labels into one-hot encoded vectors.
    Attributes:
        nClasses (int): The number of unique classes for one-hot encoding.
    """

    def __init__(self, nClasses: int):
        assert nClasses > 0, "Number of classes must be > 0."
        self.nClasses = nClasses

    def __call__(self, data: Data):
        """
        Applies one-hot encoding to the node features of the input data.

        Args:
            data (Data): A PyTorch Geometric Data object.

        Returns:
            Data: The transformed data with one-hot encoded features.
        """
        if data.x is None:
            raise ValueError("Input data must have node features (data.x).")

        # Ensure that the node features are integers for one-hot encoding
        if not torch.is_floating_point(data.x):
            data.x = one_hot(
                data.x.long(), num_classes=self.nClasses).squeeze(1)

        return data

    def __repr__(self):
        """
        Returns a string representation of the transform.
        """
        return f'{self.__class__.__name__}(nClasses={self.nClasses})'
