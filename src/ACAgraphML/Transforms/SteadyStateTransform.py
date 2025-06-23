from torch_geometric.transforms import BaseTransform, OneHotDegree
from torch_geometric.utils import to_dense_adj
import torch
from torch_geometric.data.data import Data


class SteadyStateTransform(BaseTransform):
    """
    A PyTorch Geometric transform that appends the steady-state vector of the 
    (optionally weighted) adjacency matrix to each node's feature vector.

    The steady-state vector is computed by raising the normalized adjacency 
    matrix to a high power, simulating the stationary distribution of a 
    random walk on the graph.
    """

    def __init__(self, power=500, useEdgeWeights=True):
        """
        Args:
            power (int): The exponent to which the normalized adjacency matrix is raised.
            useEdgeWeights (bool): Whether to use edge weights in the adjacency matrix.
        """
        super(SteadyStateTransform, self).__init__()
        self.useEdgeWeights = useEdgeWeights
        self.power = power

    def computeSteadyState(self, graph: Data):
        """
        Computes the steady-state vector for the given graph.

        Args:
            graph (Data): A PyTorch Geometric Data object representing the graph.

        Returns:
            torch.Tensor: The steady-state vector for each node (shape: [num_nodes, 1]).
        """
        # Convert the graph to a dense adjacency matrix (optionally weighted)
        adj = to_dense_adj(
            graph.edge_index,
            edge_attr=graph.edge_attr if self.useEdgeWeights else None
        ).squeeze(0)

        # Normalize the adjacency matrix row-wise (stochastic matrix)
        row_sums = adj.sum(dim=1, keepdim=True)
        adj = adj / row_sums.clamp(min=1e-10)  # Avoid division by zero

        # Compute the steady-state vector by raising the matrix to a high power
        steady_state = torch.linalg.matrix_power(adj, self.power)[
            0].unsqueeze(1)

        return steady_state

    def __call__(self, data: Data):
        """
        Applies the steady-state transform to the input data.

        Args:
            data (Data): A PyTorch Geometric Data object.

        Returns:
            Data: The input data with the steady-state vector appended to node features.
        """
        steadyState = self.computeSteadyState(data)
        data.x = torch.cat([data.x, steadyState], dim=-1)
        return data

    def __repr__(self):
        """
        Returns a string representation of the transform.
        """
        return f'{self.__class__.__name__}(useEdgeWeights={self.useEdgeWeights})'


if __name__ == "__main__":
    # Example usage and test
    from ACAgraphML.Dataset import ZINC_Dataset
    dataset = ZINC_Dataset.SMALL_TEST.load(
        transform=SteadyStateTransform(power=500, useEdgeWeights=True))
    print(dataset[2].x.shape)
