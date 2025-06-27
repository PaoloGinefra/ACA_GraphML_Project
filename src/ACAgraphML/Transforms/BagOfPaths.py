from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_dense_adj
import torch
from torch_geometric.data.data import Data


class BagOfPathsTransform(BaseTransform):
    def __init__(self, maxPathLength=10, minPathLength=2, logResults=False):
        super().__init__()
        assert maxPathLength >= minPathLength, "maxPathLength must be greater than or equal to minPathLength"
        self.maxPathLength = maxPathLength
        self.minPathLength = minPathLength
        self.logResults = logResults

    def __call__(self, data: Data):
        if data.edge_index is None:
            raise ValueError(
                "Input data must have edge_index (graph structure).")

        # Convert the graph to a dense adjacency matrix
        adj = to_dense_adj(data.edge_index).squeeze(0)

        # Initialize the bag of paths tensor
        num_nodes = adj.size(0)

        delta = self.maxPathLength - self.minPathLength + 1

        BOP = torch.zeros(delta, dtype=torch.float32)
        A = adj.clone()
        for i in range(self.minPathLength, self.maxPathLength + 1):
            if i > 1:
                A = torch.matmul(A, adj)
            trace = torch.trace(A)
            npaths = trace.item()/(2 * i)
            BOP[i - self.minPathLength] = npaths

        if self.logResults:
            BOP = torch.log(BOP + 1)

        data.x = BOP
        return data
