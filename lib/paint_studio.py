### This script will contain methods to draw and visualise the graph, and calculated statistics, as well as colouring community on a graph and their annotations
"""
paint_studio.py

Draft visualization utilities for regulatory network analysis.

Includes:
- Drawing network graphs with node/edge statistics
- Visualizing community structure (e.g., Leiden clusters)
- Overlaying annotations, centrality, entropy, etc.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, Optional, Sequence, Any

def draw_network(
    G: nx.Graph,
    node_stats: Optional[Dict[str, Dict[str, float]]] = None,
    community_dict: Optional[Dict[Any, int]] = None,
    node_color_metric: str = 'degree_centrality',
    cmap: str = 'viridis',
    layout: str = 'spring',
    with_labels: bool = True,
    node_size: int = 300,
    edge_alpha: float = 0.4,
    ax: Optional[plt.Axes] = None,
    **kwargs
):
    """
    Visualizes a networkx graph with optional node statistics and community colouring.

    Args:
        G: networkx graph
        node_stats: {node: {metric: value}}
        community_dict: {node: cluster_id}
        node_color_metric: which stat to color nodes by, if available
        cmap: colormap
        layout: 'spring', 'kamada_kawai', 'circular', etc.
        with_labels: display node labels
        node_size: base node size
        edge_alpha: edge transparency
        ax: matplotlib Axes

    Returns:
        matplotlib Figure, Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8,8)))
    else:
        fig = ax.figure

    # Layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=1)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=1)

    # Node coloring
    if community_dict is not None:
        communities = [community_dict.get(n, None) for n in G.nodes()]
        color_vals = communities
        num_colors = len(set(communities))
        cmap_choice = plt.cm.get_cmap(cmap, num_colors)
    elif node_stats is not None and node_color_metric in next(iter(node_stats.values())):
        color_vals = [node_stats[n][node_color_metric] for n in G.nodes()]
        cmap_choice = plt.cm.get_cmap(cmap)
    else:
        color_vals = 'gray'
        cmap_choice = None

    # Draw network
    nx.draw(
        G, pos=pos, node_color=color_vals, cmap=cmap_choice, 
        node_size=node_size, edge_color='k', alpha=edge_alpha,
        with_labels=with_labels, ax=ax
    )

    # Add colorbar if coloring by metric
    if (community_dict or (node_stats and node_color_metric in next(iter(node_stats.values())))):
        sm = plt.cm.ScalarMappable(cmap=cmap_choice)
        fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.03, aspect=30)

    ax.set_axis_off()
    plt.tight_layout()
    return fig, ax

def highlight_communities(
    G: nx.Graph,
    community_labels: Dict[Any, int],
    pos: Optional[Dict[Any, np.ndarray]] = None,
    cmap: str = "tab20",
    alpha: float = 0.7,
    with_labels: bool = False,
    **kwargs
):
    """
    Color nodes in the graph by community/cluster assignment.

    Args:
        G: networkx graph
        community_labels: {node: leiden/group id}
        pos: node positions (optional)
        cmap: color map
        alpha: node transparency
        with_labels: show node labels

    Returns:
        fig, ax
    """
    if pos is None:
        pos = nx.spring_layout(G, seed=2)
    num_communities = len(set(community_labels.values()))
    cmap_choice = plt.cm.get_cmap(cmap, num_communities)
    node_colors = [community_labels[n] for n in G.nodes()]
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8,8)))
    nx.draw(
        G, pos, node_color=node_colors, cmap=cmap_choice,
        with_labels=with_labels, edge_color="gray", alpha=alpha, ax=ax
    )
    sm = plt.cm.ScalarMappable(cmap=cmap_choice)
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.03, aspect=30)
    ax.set_axis_off()
    plt.tight_layout()
    return fig, ax

def plot_node_stat_distribution(
    node_stats: Dict[str, Dict[str, float]],
    metric: str,
    bins: int = 30,
    ax: Optional[plt.Axes] = None,
    **kwargs
):
    """
    Plot histogram of a given node-level metric.

    Args:
        node_stats: {node: {stat_name: value}}
        metric: which stat to plot (e.g., 'betweenness_centrality')
        bins: histogram bins
        ax: matplotlib Axes

    Returns:
        fig, ax
    """
    vals = [d[metric] for d in node_stats.values()]
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (6,4)))
    else:
        fig = ax.figure
    ax.hist(vals, bins=bins, color='purple', alpha=0.65)
    ax.set_title(f"{metric} distribution")
    ax.set_xlabel(metric)
    ax.set_ylabel("Count")
    plt.tight_layout()
    return fig, ax

def annotate_clusters(
    cluster_dict: Dict[int, Any],
    enrichment_results: Optional[Dict[int, Dict[str, float]]] = None,
    top_n: int = 3
) -> Dict[int, Dict[str, Any]]:
    """
    For each cluster, summarize functional annotation (e.g., from enrichment).

    Args:
        cluster_dict: {node: cluster_id}
        enrichment_results: {cluster_id: {term: score}}
        top_n: how many top terms to keep

    Returns:
        {cluster_id: {"top_terms": [(term, score), ...]}}
    """
    cluster_info = {}
    if enrichment_results is not None:
        for cid, term_scores in enrichment_results.items():
            sorted_terms = sorted(term_scores.items(), key=lambda x: -x[1])[:top_n]
            cluster_info[cid] = {"top_terms": sorted_terms}
    else:
        for cid in set(cluster_dict.values()):
            cluster_info[cid] = {"top_terms": []}
    return cluster_info

def draw_cluster_annotations(
    fig: plt.Figure, 
    ax: plt.Axes, 
    cluster_info: Dict[int, Dict[str, Any]],
    label_pos: Optional[Dict[int, Any]] = None
):
    """
    Overlay cluster annotations onto graph plot.

    Args:
        fig, ax: matplotlib objects
        cluster_info: {cluster_id: {"top_terms": [(term, score), ...]}}
        label_pos: {cluster_id: (x, y)} optional

    """
    for cid, info in cluster_info.items():
        top_terms = info.get("top_terms", [])
        if label_pos and cid in label_pos:
            xy = label_pos[cid]
        else:
            xy = (0.1 + 0.8*np.random.rand(), 0.1 + 0.8*np.random.rand())
        label = ", ".join(t for t, _ in top_terms)
        ax.text(*xy, f"Cluster {cid}\n{label}", bbox=dict(facecolor="white", alpha=0.6), fontsize=9, ha='center')




def plot_graph_with_threshold(G: nx.Graph, threshold: float = 0.4, output_file: str = 'GRN_network.png'):
    """
        Plot the graph with edges above threshold highlighted.

        Args:
        G: NetworkX graph
        threshold: Weight threshold for highlighting edges
        output_file: Output filename for the plot
    """
    plt.figure(figsize=(10, 10))
    
    # Get edge weights
    edge_weights = nx.get_edge_attributes(G, 'weight')
    
    # Separate edges by threshold
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d.get("weight", 0) > threshold]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d.get("weight", 0) <= threshold]
    
    # Calculate layout
    pos = nx.spring_layout(G, k=0.4, iterations=15, seed=3)
    
    # Prepare edge widths
    width_large = {edge: edge_weights.get(edge, 0) * 10 for edge in elarge}
    width_small = {edge: max(edge_weights.get(edge, 0), 0) * 10 for edge in esmall}
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        edgelist=list(width_small.keys()),
        width=list(width_small.values()),
        edge_color='lightblue',
        alpha=0.8
    )
    nx.draw_networkx_edges(
        G, pos,
        edgelist=list(width_large.keys()),
        width=list(width_large.values()),
        alpha=0.5,
        edge_color="blue",
    )
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
    
    # Draw edge weight labels for significant edges
    edge_labels = {edge: edge_weights.get(edge, 0) for edge in elarge}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=15)
    
    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_top_nodes(df, metric_column, top_n=10, ax=None, title=None):
    """
    Plot top N nodes for a given centrality metric.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with node names as index and centrality metrics as columns
    metric_column : str
        Name of the column to rank and plot
    top_n : int
        Number of top nodes to display (default: 10)
    ax : matplotlib.axes.Axes
        Axes object to plot on. If None, creates new figure
    title : str
        Custom title for the plot. If None, uses metric_column name
    
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes object with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))
    
    # Convert column to numeric, handling errors and non-numeric values
    if metric_column not in df.columns:
        ax.text(0.5, 0.5, f'Column "{metric_column}" not found', 
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Convert to numeric, coercing errors to NaN
    numeric_series = pd.to_numeric(df[metric_column], errors='coerce')
    
    # Check if we have any valid numeric values
    if numeric_series.isna().all():
        ax.text(0.5, 0.5, f'No valid numeric values in "{metric_column}"', 
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Create a temporary dataframe with numeric values for sorting
    df_numeric = df.copy()
    df_numeric[metric_column] = numeric_series
    
    # Get top N nodes (drop NaN values first)
    df_valid = df_numeric.dropna(subset=[metric_column])
    if len(df_valid) == 0:
        ax.text(0.5, 0.5, f'No valid values in "{metric_column}"', 
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    top_nodes = df_valid.nlargest(top_n, metric_column)
    
    # Reverse order so highest is on top
    top_nodes = top_nodes.iloc[::-1]
    
    # Create y positions
    y_pos = np.arange(len(top_nodes))
    
    # Plot
    ax.scatter(top_nodes[metric_column], y_pos, s=80, alpha=0.8, color='#1f77b4')
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_nodes.index)
    ax.set_xlabel(metric_column.replace('_', ' ').title())
    
    if title:
        ax.set_title(title, fontsize=11, pad=10)
    else:
        ax.set_title(f'{metric_column.replace("_", " ").title()}\ntop {top_n}', 
                    fontsize=11, pad=10)
    
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return ax


def plot_multiple_metrics(df, metrics, top_n=10, figsize=(12, 8), ncols=2, out_path=None):
    """
    Create a grid of plots for multiple centrality metrics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with node names as index and centrality metrics as columns
    metrics : list of str
        List of column names to plot
    top_n : int
        Number of top nodes to display per metric (default: 10)
    figsize : tuple
        Figure size (width, height)
    ncols : int
        Number of columns in the subplot grid
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    axes : array of matplotlib.axes.Axes
        Array of axes objects
    """
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Flatten axes array for easier iteration
    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    last_used_axis = -1
    for i, metric in enumerate(metrics):
        # Skip if metric column doesn't exist
        if metric not in df.columns:
            axes[i].text(0.5, 0.5, f'Column "{metric}" not found', 
                        ha='center', va='center', transform=axes[i].transAxes)
            last_used_axis = i
            continue
        
        # Skip if column is not numeric (after conversion attempt)
        numeric_series = pd.to_numeric(df[metric], errors='coerce')
        if numeric_series.isna().all():
            axes[i].text(0.5, 0.5, f'No valid numeric values in "{metric}"', 
                        ha='center', va='center', transform=axes[i].transAxes)
            last_used_axis = i
            continue
        
        plot_top_nodes(df, metric, top_n=top_n, ax=axes[i])
        last_used_axis = i
    
    # Hide empty subplots
    for j in range(last_used_axis + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    else:
        return fig, axes
