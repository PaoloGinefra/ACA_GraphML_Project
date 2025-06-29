import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data.data import Data


class AddMasterNode(BaseTransform):
    """
    A PyTorch Geometric transform that adds a master node to a graph.
    The master node is connected to all other nodes in the graph (bidirectionally).
    This can be useful for global information aggregation in graph neural networks.

    The master node features can be either zero-initialized, mean-initialized,
    or set to a specific value.

    Attributes:
        node_feature_init (str): How to initialize the master node features.
                                Options: 'zero', 'mean', 'ones'
        edge_feature_init (str): How to initialize the edge features for connections
                                to/from the master node. Options: 'zero', 'mean', 'ones'
    """

    def __init__(self, node_feature_init: str = 'zero', edge_feature_init: str = 'zero'):
        """
        Initialize the AddMasterNode transform.

        Args:
            node_feature_init (str): Initialization strategy for master node features.
                                   'zero': all zeros, 'mean': mean of existing nodes, 'ones': all ones
            edge_feature_init (str): Initialization strategy for master node edge features.
                                   'zero': all zeros, 'mean': mean of existing edges, 'ones': all ones
        """
        assert node_feature_init in ['zero', 'mean', 'ones'], \
            "node_feature_init must be one of: 'zero', 'mean', 'ones'"
        assert edge_feature_init in ['zero', 'mean', 'ones'], \
            "edge_feature_init must be one of: 'zero', 'mean', 'ones'"

        self.node_feature_init = node_feature_init
        self.edge_feature_init = edge_feature_init

    def __call__(self, data: Data) -> Data:
        """
        Apply the transform to add a master node to the graph.

        Args:
            data (Data): A PyTorch Geometric Data object.

        Returns:
            Data: The transformed data object with an added master node.
        """
        # Get the number of nodes in the original graph
        num_nodes = data.x.size(0)
        master_node_idx = num_nodes  # Index of the new master node

        # Create master node features
        if self.node_feature_init == 'zero':
            master_node_features = torch.zeros(
                1, data.x.size(1), dtype=data.x.dtype)
        elif self.node_feature_init == 'mean':
            master_node_features = data.x.float().mean(dim=0, keepdim=True)
        elif self.node_feature_init == 'ones':
            master_node_features = torch.ones(
                1, data.x.size(1), dtype=data.x.dtype)

        # Add master node to node features
        if self.node_feature_init == 'mean':
            data.x = torch.cat([data.x.float(), master_node_features], dim=0)
        else:
            data.x = torch.cat([data.x, master_node_features], dim=0)

        # Create edges: master node connects to all other nodes (bidirectionally)
        # Edges from master to all other nodes
        master_to_others = torch.stack([
            torch.full((num_nodes,), master_node_idx, dtype=torch.long),
            torch.arange(num_nodes, dtype=torch.long)
        ])

        # Edges from all other nodes to master
        others_to_master = torch.stack([
            torch.arange(num_nodes, dtype=torch.long),
            torch.full((num_nodes,), master_node_idx, dtype=torch.long)
        ])

        # Combine all edges
        new_edges = torch.cat([master_to_others, others_to_master], dim=1)
        data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)

        # Handle edge attributes if they exist
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            num_new_edges = new_edges.size(1)
            if data.edge_attr.dim() == 1:
                # If edge_attr is 1D, reshape to 2D
                data.edge_attr = data.edge_attr.unsqueeze(-1)

            if self.edge_feature_init == 'zero':
                new_edge_features = torch.zeros(num_new_edges, data.edge_attr.size(1),
                                                dtype=data.edge_attr.dtype)
            elif self.edge_feature_init == 'mean':
                new_edge_features = data.edge_attr.mean(
                    dim=0, keepdim=True).repeat(num_new_edges, 1)
            elif self.edge_feature_init == 'ones':
                new_edge_features = torch.ones(num_new_edges, data.edge_attr.size(1),
                                               dtype=data.edge_attr.dtype)

            data.edge_attr = torch.cat(
                [data.edge_attr, new_edge_features], dim=0)

        # Update batch information if it exists (for batched graphs)
        if hasattr(data, 'batch') and data.batch is not None:
            # Assign the master node to the same batch as the last node
            last_batch_idx = data.batch[-1]
            master_batch = torch.tensor(
                [last_batch_idx], dtype=data.batch.dtype)
            data.batch = torch.cat([data.batch, master_batch])

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'node_feature_init={self.node_feature_init}, '
                f'edge_feature_init={self.edge_feature_init})')
