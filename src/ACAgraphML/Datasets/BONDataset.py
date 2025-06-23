from torch.utils.data import Dataset
from torch_geometric.data import Dataset as gDataset
from torch.nn.functional import one_hot
from ..Transforms import SteadyStateTransform
import torch


class BONDataset(Dataset):
    """
    A PyTorch Dataset wrapper for a graph dataset, providing additional
    features such as one-hot node/edge encodings, BON (Bag of Nodes) features,
    edge attribute summaries, and steady-state node features.

    Args:
        graphDataset (gDataset): A PyTorch Geometric dataset containing graphs.
    """

    def __init__(self, graphDataset: gDataset):
        """
        Initializes the BONDataset by precomputing node and edge features,
        as well as steady-state features for each graph in the dataset.
        """
        super().__init__()
        # Number of unique node features (assuming categorical node features)
        self.nNodeFeats = torch.max(
            graphDataset.x, dim=0).values.item() + 1

        # Number of unique edge features (assuming categorical edge features)
        self.nEdgeFeats = torch.max(
            graphDataset.edge_attr, dim=0).values.item() + 1

        self.graphDataset = graphDataset

        # One-hot encode node features for each graph
        self.oneHots = [one_hot(
            graph.x, num_classes=self.nNodeFeats).squeeze() for graph in graphDataset]

        # Bag of Nodes (BON): sum of one-hot encodings for each graph
        self.BON = [onehot.sum(0) for onehot in self.oneHots]
        self.BON = torch.stack(self.BON, dim=0)

        # One-hot encode edge attributes, sum over edges, and remove first column (often background)
        self.edgeAttrs = [one_hot(
            graph.edge_attr, num_classes=self.nEdgeFeats).squeeze(dim=1).sum(dim=0) for graph in graphDataset]
        self.edgeAttrs = torch.stack(self.edgeAttrs, dim=0)[:, 1:]

        # Compute steady-state node features for each graph (using personalized PageRank or similar)
        self.steadyStates = [
            SteadyStateTransform(useEdgeWeights=True)(graph).x[:, [1]] for graph in graphDataset
        ]

        # Print shapes for debugging
        print(self.oneHots[0].shape, self.steadyStates[0].shape)

        # Weighted BON: node one-hots weighted by steady-state values, summed for each graph
        self.steadyBON = [(onehot * steady).sum(0) for onehot,
                          steady in zip(self.oneHots, self.steadyStates)]

    def __len__(self):
        """
        Returns the number of graphs in the dataset.
        """
        return len(self.graphDataset)

    def __getitem__(self, idx):
        """
        Returns the features and label for the graph at the given index.

        Returns:
            tuple: (BON, steadyBON, edgeAttrs, label)
                - BON: Bag of Nodes feature vector (tensor)
                - steadyBON: Steady-state weighted BON feature vector (tensor)
                - edgeAttrs: Summed edge attribute one-hot vector (tensor)
                - label: Graph label (tensor)
        """
        return self.BON[idx], self.steadyBON[idx], self.edgeAttrs[idx], self.graphDataset[idx].y
