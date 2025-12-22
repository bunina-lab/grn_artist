### This script will contain methods to draw and visualise the graph, and calculated statistics, as well as colouring community on a graph and their annotations
"""
paint_studio.py

Draft visualization utilities for regulatory network analysis.

Includes:
- Drawing network graphs with node/edge statistics
- Visualizing community structure (e.g., Leiden clusters)
- Overlaying annotations, centrality, entropy, etc.
"""

import networkx as nx
from typing import Dict, Optional, Sequence, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 600

class NetworkStatsVisualizer:
    """
    Comprehensive visualization suite for observed vs null network statistics
    """
    
    def __init__(self, obs_null_stats_list, simulations_dist_dict):
        """
        Parameters:
        -----------
        obs_null_stats_list : list of dict
            Each dict contains: metric, obs_val, sim_mean, sim_std, z_score, p_val, percentile_pos
        simulations_dist_dict : dict
            Keys are metric names, values are lists of simulated values
        """
        self.stats_df = pd.DataFrame(obs_null_stats_list)
        self.simulations = simulations_dist_dict
        
    def plot_single_metric_distribution(self, metric_name, figsize=(10, 6), 
                                       show_percentiles=True, bins=50):
        """
        Plot 1: Distribution histogram with observed value marked
        
        Best for: Understanding where your observed value falls in null distribution
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get data
        dist = self.simulations[metric_name]
        row = self.stats_df[self.stats_df['metric'] == metric_name].iloc[0]
        obs_val = row['obs_val']
        p_val = row['p_val']
        z_score = row['z_score']
        
        # Plot histogram
        counts, bins_edges, patches = ax.hist(dist, bins=bins, alpha=0.7, 
                                              color='skyblue', edgecolor='black')
        
        # Mark observed value
        ax.axvline(obs_val, color='red', linewidth=2.5, 
                  label=f'Observed (z={z_score:.2f}, p={p_val:.4f})')
        
        # Mark mean and confidence intervals
        mean_val = np.mean(dist)
        std_val = np.std(dist)
        ax.axvline(mean_val, color='gray', linestyle='--', linewidth=1.5, 
                  label='Null Mean')
        
        if show_percentiles:
            p5, p95 = np.percentile(dist, [5, 95])
            ax.axvspan(p5, p95, alpha=0.2, color='green', 
                      label='90% CI')
        
        # Shade significance regions (p < 0.05)
        z_critical = 1.96  # Two-tailed, alpha=0.05
        lower_critical = mean_val - z_critical * std_val
        upper_critical = mean_val + z_critical * std_val
        
        # Shade left tail
        if obs_val < lower_critical:
            ax.axvspan(min(dist), lower_critical, alpha=0.15, color='red')
        # Shade right tail
        if obs_val > upper_critical:
            ax.axvspan(upper_critical, max(dist), alpha=0.15, color='red')
        
        ax.set_xlabel(metric_name.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Null Distribution vs Observed: {metric_name}', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_z_score_forest(self, figsize=(10, 8), significance_level=0.05):
        """
        Plot 2: Forest plot of z-scores
        
        Best for: Comparing effect sizes across all metrics at once
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort by absolute z-score
        df_sorted = self.stats_df.copy()
        df_sorted['abs_z'] = df_sorted['z_score'].abs()
        df_sorted = df_sorted.sort_values('abs_z', ascending=True)
        
        # Calculate significance threshold
        z_critical = stats.norm.ppf(1 - significance_level/2)
        
        # Create colors based on significance
        colors = ['red' if abs(z) > z_critical else 'gray' 
                 for z in df_sorted['z_score']]
        
        # Plot horizontal bars
        y_pos = np.arange(len(df_sorted))
        ax.barh(y_pos, df_sorted['z_score'], color=colors, alpha=0.7, 
               edgecolor='black')
        
        # Add significance threshold lines
        ax.axvline(z_critical, color='red', linestyle='--', linewidth=1.5, 
                  alpha=0.5, label=f'α={significance_level}')
        ax.axvline(-z_critical, color='red', linestyle='--', linewidth=1.5, 
                  alpha=0.5)
        ax.axvline(0, color='black', linewidth=1)
        
        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sorted['metric'].str.replace('_', ' '), fontsize=10)
        ax.set_xlabel('Z-Score', fontsize=12)
        ax.set_title('Z-Scores: Observed vs Null Model', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    def plot_p_value_heatmap(self, figsize=(12, 6)):
        """
        Plot 3: P-value heatmap with effect direction
        
        Best for: Quick overview of significance and direction
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, 
                                       gridspec_kw={'width_ratios': [3, 1]})
        
        # Prepare data
        df_sorted = self.stats_df.sort_values('p_val')
        
        # Create signed log p-value (shows direction and significance)
        df_sorted['signed_log_p'] = -np.log10(df_sorted['p_val']) * np.sign(df_sorted['z_score'])
        
        # Heatmap
        metrics = df_sorted['metric'].values
        values = df_sorted['signed_log_p'].values.reshape(-1, 1)
        
        im = ax1.imshow(values, cmap='RdBu_r', aspect='auto', 
                       vmin=-5, vmax=5)
        
        ax1.set_yticks(np.arange(len(metrics)))
        ax1.set_yticklabels(metrics, fontsize=10)
        ax1.set_xticks([0])
        ax1.set_xticklabels(['Signed -log10(p)'])
        ax1.set_title('Significance & Direction', fontsize=12, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Signed -log10(p-value)', rotation=270, labelpad=20)
        
        # Add significance markers
        for i, (p, z) in enumerate(zip(df_sorted['p_val'], df_sorted['z_score'])):
            if p < 0.001:
                marker = '***'
            elif p < 0.01:
                marker = '**'
            elif p < 0.05:
                marker = '*'
            else:
                marker = ''
            ax1.text(0, i, marker, ha='center', va='center', 
                    color='black', fontweight='bold', fontsize=14)
        
        # Bar plot of p-values
        colors = ['red' if p < 0.05 else 'gray' for p in df_sorted['p_val']]
        ax2.barh(np.arange(len(metrics)), -np.log10(df_sorted['p_val']), 
                color=colors, alpha=0.7, edgecolor='black')
        ax2.axvline(-np.log10(0.05), color='red', linestyle='--', 
                   linewidth=2, label='p=0.05')
        ax2.set_xlabel('-log10(p-value)', fontsize=10)
        ax2.set_yticks([])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    def plot_percentile_positions(self, figsize=(10, 8)):
        """
        Plot 4: Percentile position plot
        
        Best for: Understanding how extreme your observations are
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        df_sorted = self.stats_df.sort_values('percentile_pos', ascending=True)
        
        y_pos = np.arange(len(df_sorted))
        percentiles = df_sorted['percentile_pos'].values
        
        # Create color gradient based on distance from 50th percentile
        colors = plt.cm.RdYlBu_r((percentiles - 50) / 50)
        
        ax.barh(y_pos, percentiles, color=colors, alpha=0.8, edgecolor='black')
        
        # Add reference lines
        ax.axvline(50, color='gray', linestyle='-', linewidth=2, 
                  label='Median (50th)')
        ax.axvline(5, color='red', linestyle='--', linewidth=1.5, 
                  alpha=0.7, label='5th/95th percentile')
        ax.axvline(95, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sorted['metric'].str.replace('_', ' '), fontsize=10)
        ax.set_xlabel('Percentile Position in Null Distribution', fontsize=12)
        ax.set_xlim([0, 100])
        ax.set_title('Percentile Position of Observed Values', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    def plot_volcano(self, figsize=(10, 8), p_threshold=0.05, z_threshold=1.96):
        """
        Plot 5: Volcano plot (effect size vs significance)
        
        Best for: Identifying metrics with large AND significant deviations
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        z_scores = self.stats_df['z_score'].values
        neg_log_p = -np.log10(self.stats_df['p_val'].values)
        
        # Color code points
        colors = []
        for z, p in zip(z_scores, self.stats_df['p_val']):
            if p < p_threshold and abs(z) > z_threshold:
                colors.append('red')  # Significant
            else:
                colors.append('gray')  # Not significant
        
        # Scatter plot
        ax.scatter(z_scores, neg_log_p, c=colors, alpha=0.7, s=100, 
                  edgecolors='black', linewidth=1.5)
        
        # Add threshold lines
        ax.axhline(-np.log10(p_threshold), color='blue', linestyle='--', 
                  linewidth=2, label=f'p={p_threshold}')
        ax.axvline(z_threshold, color='green', linestyle='--', linewidth=2, 
                  alpha=0.5, label=f'z=±{z_threshold}')
        ax.axvline(-z_threshold, color='green', linestyle='--', linewidth=2, 
                  alpha=0.5)
        
        # Label significant points
        for idx, row in self.stats_df.iterrows():
            if row['p_val'] < p_threshold and abs(row['z_score']) > z_threshold:
                ax.annotate(row['metric'], 
                          xy=(row['z_score'], -np.log10(row['p_val'])),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=9, alpha=0.8,
                          bbox=dict(boxstyle='round,pad=0.3', 
                                  facecolor='yellow', alpha=0.5))
        
        ax.set_xlabel('Z-Score (Effect Size)', fontsize=12)
        ax.set_ylabel('-log10(p-value)', fontsize=12)
        ax.set_title('Volcano Plot: Effect Size vs Significance', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_multi_metric_grid(self, metrics_list=None, ncols=3, 
                               figsize_per_plot=(6, 4), bins=40):
        """
        Plot 6: Small multiples - grid of distribution plots
        
        Best for: Detailed examination of multiple metrics simultaneously
        """
        if metrics_list is None:
            metrics_list = self.stats_df['metric'].tolist()
        
        nrows = int(np.ceil(len(metrics_list) / ncols))
        fig, axes = plt.subplots(nrows, ncols, 
                                figsize=(figsize_per_plot[0]*ncols, 
                                        figsize_per_plot[1]*nrows))
        axes = axes.flatten() if nrows > 1 else [axes]
        
        for idx, metric in enumerate(metrics_list):
            ax = axes[idx]
            
            # Get data
            dist = self.simulations[metric]
            row = self.stats_df[self.stats_df['metric'] == metric].iloc[0]
            obs_val = row['obs_val']
            p_val = row['p_val']
            
            # Convert to numpy array and remove NaN/Inf values
            dist_array = np.array(dist)
            dist_array = dist_array[np.isfinite(dist_array)]
            
            if len(dist_array) == 0:
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(metric.replace('_', ' ').title(), fontsize=10, 
                            fontweight='bold')
                continue
            
            # Calculate appropriate number of bins based on data
            data_range = np.max(dist_array) - np.min(dist_array)
            n_unique = len(np.unique(dist_array))
            
            # Use the minimum of: requested bins, number of unique values, or bins based on data range
            # If all values are identical or range is 0, use 1 bin
            if data_range == 0 or n_unique == 1:
                actual_bins = 1
            else:
                # Use fewer bins if we have fewer unique values
                # Also check if range is too small for requested bins
                actual_bins = min(bins, n_unique, max(1, int(len(dist_array) ** 0.5)))
            
            # Plot
            ax.hist(dist_array, bins=actual_bins, alpha=0.6, color='skyblue', 
                   edgecolor='black')
            if np.isfinite(obs_val):
                ax.axvline(obs_val, color='red', linewidth=2, 
                          label=f'Obs (p={p_val:.3f})')
            mean_val = np.mean(dist_array)
            if np.isfinite(mean_val):
                ax.axvline(mean_val, color='gray', linestyle='--', 
                          linewidth=1.5)
            
            # Styling
            ax.set_title(metric.replace('_', ' ').title(), fontsize=10, 
                        fontweight='bold')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
        
        # Hide empty subplots
        for idx in range(len(metrics_list), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_qq_plot(self, metric_name, figsize=(8, 8)):
        """
        Plot 7: Q-Q plot for normality check
        
        Best for: Checking if null distribution is approximately normal
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        dist = self.simulations[metric_name]
        
        # Standardize
        standardized = (dist - np.mean(dist)) / np.std(dist)
        
        # Q-Q plot
        stats.probplot(standardized, dist="norm", plot=ax)
        
        ax.set_title(f'Q-Q Plot: {metric_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_summary_report(self, output_path='network_stats_report.png', 
                             dpi=300):
        """
        Plot 8: Comprehensive summary figure combining key visualizations
        
        Best for: Publication-ready overview of all results
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Forest plot (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        df_sorted = self.stats_df.copy()
        df_sorted['abs_z'] = df_sorted['z_score'].abs()
        df_sorted = df_sorted.sort_values('abs_z', ascending=True).tail(10)
        
        z_critical = 1.96
        colors = ['red' if abs(z) > z_critical else 'gray' 
                 for z in df_sorted['z_score']]
        y_pos = np.arange(len(df_sorted))
        ax1.barh(y_pos, df_sorted['z_score'], color=colors, alpha=0.7)
        ax1.axvline(z_critical, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax1.axvline(-z_critical, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(df_sorted['metric'].str.replace('_', ' '), fontsize=9)
        ax1.set_xlabel('Z-Score')
        ax1.set_title('Top 10 Z-Scores', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 2. Volcano plot (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        z_scores = self.stats_df['z_score'].values
        neg_log_p = -np.log10(self.stats_df['p_val'].values)
        colors = ['red' if p < 0.05 and abs(z) > 1.96 else 'gray' 
                 for z, p in zip(z_scores, self.stats_df['p_val'])]
        ax2.scatter(z_scores, neg_log_p, c=colors, alpha=0.7, s=80)
        ax2.axhline(-np.log10(0.05), color='blue', linestyle='--', linewidth=2)
        ax2.axvline(1.96, color='green', linestyle='--', linewidth=2, alpha=0.5)
        ax2.axvline(-1.96, color='green', linestyle='--', linewidth=2, alpha=0.5)
        ax2.set_xlabel('Z-Score')
        ax2.set_ylabel('-log10(p-value)')
        ax2.set_title('Volcano Plot', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. P-value distribution (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(self.stats_df['p_val'], bins=20, alpha=0.7, 
                color='skyblue', edgecolor='black')
        ax3.axvline(0.05, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('P-value')
        ax3.set_ylabel('Count')
        ax3.set_title('P-value Distribution', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Percentile positions (middle right)
        ax4 = fig.add_subplot(gs[1, 1])
        df_extreme = self.stats_df[
            (self.stats_df['percentile_pos'] < 5) | 
            (self.stats_df['percentile_pos'] > 95)
        ].sort_values('percentile_pos')
        if len(df_extreme) > 0:
            y_pos = np.arange(len(df_extreme))
            colors = plt.cm.RdYlBu_r((df_extreme['percentile_pos'] - 50) / 50)
            ax4.barh(y_pos, df_extreme['percentile_pos'], color=colors, alpha=0.8)
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(df_extreme['metric'].str.replace('_', ' '), fontsize=9)
            ax4.axvline(50, color='gray', linestyle='-', linewidth=2)
            ax4.set_xlabel('Percentile')
            ax4.set_title('Extreme Percentile Positions', fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='x')
        
        # 5. Example distribution (bottom, spanning both columns)
        ax5 = fig.add_subplot(gs[2, :])
        # Pick most significant metric
        most_sig = self.stats_df.nsmallest(1, 'p_val').iloc[0]
        metric_name = most_sig['metric']
        dist = self.simulations[metric_name]
        obs_val = most_sig['obs_val']
        
        ax5.hist(dist, bins=50, alpha=0.6, color='skyblue', edgecolor='black')
        ax5.axvline(obs_val, color='red', linewidth=2.5, 
                   label=f'Observed (p={most_sig["p_val"]:.4f})')
        ax5.axvline(np.mean(dist), color='gray', linestyle='--', linewidth=1.5, 
                   label='Null Mean')
        ax5.set_xlabel(metric_name.replace('_', ' ').title())
        ax5.set_ylabel('Frequency')
        ax5.set_title(f'Most Significant Metric: {metric_name}', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        plt.suptitle('Network Statistics: Observed vs Null Model Summary', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        return fig


# Example usage
if __name__ == "__main__":
    # Simulate example data
    np.random.seed(42)
    
    # Create fake simulation distributions
    metrics = ['density', 'transitivity', 'avg_clustering', 'assortativity', 
               'avg_degree', 'n_triangles', 'global_efficiency', 
               'shannon_entropy']
    
    simulations_dist_dict = {}
    obs_null_stats_list = []
    
    for metric in metrics:
        # Generate null distribution
        null_dist = np.random.normal(0.5, 0.1, 10000)
        simulations_dist_dict[metric] = null_dist
        
        # Generate observed value (sometimes extreme)
        if np.random.random() > 0.6:
            obs_val = np.random.normal(0.7, 0.05)  # Extreme
        else:
            obs_val = np.random.normal(0.5, 0.1)  # Normal
        
        # Calculate stats
        mean_val = np.mean(null_dist)
        std_val = np.std(null_dist)
        z_score = (obs_val - mean_val) / std_val
        p_val = 2 * (1 - stats.norm.cdf(abs(z_score)))
        percentile = stats.percentileofscore(null_dist, obs_val)
        
        obs_null_stats_list.append({
            'metric': metric,
            'obs_val': obs_val,
            'sim_mean': mean_val,
            'sim_std': std_val,
            'z_score': z_score,
            'p_val': p_val,
            'percentile_pos': percentile
        })
    
    

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
    import scipy.stats as stats

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
    
    # Statistical test: compare top N nodes to the rest of the nodes in the metric
    rest_nodes = df_valid.drop(top_nodes.index)
    significance_star = ""
    p_value = None
    if len(rest_nodes) > 1 and len(top_nodes) > 1:
        # Use nonparametric test as default (Mann-Whitney U)
        try:
            u_stat, p_value = stats.mannwhitneyu(top_nodes[metric_column], rest_nodes[metric_column], alternative='two-sided')
        except Exception:
            p_value = None
        if p_value is not None:
            if p_value < 0.001:
                significance_star = '***'
            elif p_value < 0.01:
                significance_star = '**'
            elif p_value < 0.05:
                significance_star = '*'
            else:
                significance_star = ''
    else:
        significance_star = ''
        p_value = None
    
    # Reverse order so highest is on top
    top_nodes = top_nodes.iloc[::-1]
    
    # Create y positions
    y_pos = np.arange(len(top_nodes))
    
    # Plot
    ax.scatter(top_nodes[metric_column], y_pos, s=80, alpha=0.8, color='#1f77b4')
    
    # Add significance stars next to top node labels if significant
    for i, (idx, row) in enumerate(top_nodes.iterrows()):
        # Only add star to the highest value if significant
        if i == len(top_nodes) - 1 and significance_star:
            ax.text(
                row[metric_column], y_pos[i],
                f" {significance_star}", va='center', ha='left', color='red', fontsize=17
            )

    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_nodes.index)
    ax.set_xlabel(metric_column.replace('_', ' ').title())
    
    # Update title with significance
    if title:
        auto_title = title
    else:
        auto_title = f'{metric_column.replace("_", " ").title()}\ntop {top_n}'
    if significance_star:
        auto_title += f' ({significance_star} p={p_value:.2e})'
    ax.set_title(auto_title, fontsize=11, pad=10)
    
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
