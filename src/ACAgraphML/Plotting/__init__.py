import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt


def plotGraph(graph, title='', minNodeFeat=0, maxNodeFeat=20, minEdgeFeat=1, maxEdgeFeat=3, ax: plt.Axes = None, getNodeColors=lambda g: g.x, nodesCmap=plt.cm.tab20, colorBar=False):
    G = to_networkx(graph, to_undirected=True)
    pos = nx.spring_layout(
        G, seed=42, weight=graph.edge_attr*1000, scale=2, k=2, iterations=10000)
    # Use the provided ax or create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        show_plot = True
    else:
        show_plot = False

    edge_labels = {
        (u, v): f"{graph.edge_attr[i].item()}" for i, (u, v) in enumerate(G.edges)}
    edge_colors = [graph.edge_attr[i].item() for i in range(len(G.edges))]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                           edge_cmap=plt.cm.cool, edge_vmin=minEdgeFeat, edge_vmax=maxEdgeFeat, ax=ax)
    # nx.draw_networkx_edge_labels(
    #     G, pos, edge_labels=edge_labels, font_color='black', font_size=8, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color=getNodeColors(graph),
                           cmap=nodesCmap, vmin=minNodeFeat, vmax=maxNodeFeat, ax=ax)

    labels = {i: f"{graph.x[i].item():.1f}" for i in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels,
                            font_size=8, font_color='white', ax=ax)
    ax.set_title(title)
    ax.axis('off')

    if colorBar:
        vmin = vmin if minNodeFeat else min(getNodeColors(graph))
        vmax = vmax if maxNodeFeat else max(getNodeColors(graph))
        sm = plt.cm.ScalarMappable(cmap=nodesCmap, norm=plt.Normalize(
            vmin=vmin, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Node Feature')
    if show_plot:
        plt.show()


def plotSteadyState(graph, steadyState, ax=None):
    plotGraph(graph,
              getNodeColors=lambda x: steadyState.cpu().numpy(),
              minNodeFeat=None,
              maxNodeFeat=None,
              nodesCmap='winter',
              ax=ax,
              colorBar=True)
