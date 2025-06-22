import networkx as nx
from torch_geometric.utils import to_networkx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import InMemoryDataset
import torch


def plotGraph(graph, minNodeFeat=0, maxNodeFeat=20, minEdgeFeat=1, maxEdgeFeat=3):
    G = to_networkx(graph, to_undirected=True)
    pos = nx.spring_layout(G, seed=42, weight=graph.edge_attr)

    plt.figure(figsize=(6, 6))
    edge_labels = {
        (u, v): f"{graph.edge_attr[i].item()}" for i, (u, v) in enumerate(G.edges)}
    edge_colors = [graph.edge_attr[i].item() for i in range(len(G.edges))]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                           edge_cmap=plt.cm.cool, edge_vmin=minEdgeFeat, edge_vmax=maxEdgeFeat)
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_color='black', font_size=8)
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color=graph.x,
                           cmap=plt.cm.viridis, vmin=minNodeFeat, vmax=maxNodeFeat)

    labels = {i: str(graph.x[i].item()) for i in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels,
                            font_size=8, font_color='white')
    plt.title('Graph Visualization')
    plt.axis('off')
    plt.show()
