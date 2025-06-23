from torch.utils.data import Dataset
from torch_geometric.data import Dataset as gDataset
from torch.nn.functional import one_hot
from ..Transforms import SteadyStateTransform
import torch


class BONDataset(Dataset):
    def __init__(self, graphDataset: gDataset):
        super().__init__()
        self.nNodeFeats = torch.max(
            graphDataset.x, dim=0).values.item() + 1

        self.nEdgeFeats = torch.max(
            graphDataset.edge_attr, dim=0).values.item() + 1

        self.graphDataset = graphDataset

        self.oneHots = [one_hot(
            graph.x, num_classes=self.nNodeFeats).squeeze() for graph in graphDataset]

        self.BON = [onehot.sum(0) for onehot in self.oneHots]
        self.BON = torch.stack(self.BON, dim=0)

        self.edgeAttrs = [one_hot(
            graph.edge_attr, num_classes=self.nEdgeFeats).squeeze(dim=1).sum(dim=0) for graph in graphDataset]
        self.edgeAttrs = torch.stack(self.edgeAttrs, dim=0)[:, 1:]

        self.steadyStates = [
            SteadyStateTransform(useEdgeWeights=True)(graph).x[:, [1]] for graph in graphDataset
        ]

        print(self.oneHots[0].shape, self.steadyStates[0].shape)

        self.steadyBON = [(onehot * steady).sum(0) for onehot,
                          steady in zip(self.oneHots, self.steadyStates)]

    def __len__(self):
        return len(self.graphDataset)

    def __getitem__(self, idx):
        return self.BON[idx], self.steadyBON[idx], self.edgeAttrs[idx], self.graphDataset[idx].y
