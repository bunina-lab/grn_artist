import json
import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict, List, Callable, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from lib.utils import find_files, read_json, write_to_json


def compare_to_null_model(G, metric_func, n_simulations=1000):
    """Compare network metric to random null model"""
    
    # Observed value
    observed = metric_func(G)
    
    # Generate null distribution
    null_values = []
    for _ in range(n_simulations):
        G_random = nx.gnm_random_graph(G.number_of_nodes(), 
                                       G.number_of_edges())
        null_values.append(metric_func(G_random))
    
    # Calculate statistics
    mean_null = np.mean(null_values)
    std_null = np.std(null_values)
    z_score = (observed - mean_null) / std_null
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    return {
        'observed': observed,
        'null_mean': mean_null,
        'null_std': std_null,
        'z_score': z_score,
        'p_value': p_value
    }


class GraphMetricsComparator:
    
    ACCEPTED_KEYS= [
        'n_edges', 
        'n_nodes', 
        'avg_in_degree', 
        'avg_out_degree', 
        'avg_degree', 
        'avg_closeness_centrality', 
        'avg_degree_centrality', 
        'avg_betweenness_centrality', 
        'avg_eigenvector_centrality', 
        'avg_pagerank_score', 
        'avg_eccentricity', 
        'diameter', 
        'density', 
        'transitivity', 
        'n_isolate_subgraphs', 
        'n_triangles', 
        'degree_assortativity', 
        'degree_centralization', 
        'average_clustering_coeff', 
        'global_efficiency', 
        'shannon_vertex_entropy', 
        'structural_entropy', 
        'von_neumann_entropy', 
        'shannon_degree_centrality_entropy', 
        'shannon_betweenness_centrality_entropy', 
        'shannon_pagerank_centrality_entropy', 
        'proxy_criticality_branching_ratio', 
        'proxy_average_sensitivity', 
        'girth', 
        'undireced_n_egdes', 
        'n_self_loops', 
        'n_communities'
        ]

    def __init__(self, observed_stats_json, out_dir, simulations_dir=None) -> None:

        self.observed_stats_json = observed_stats_json
        self.simulations_dir = simulations_dir
        self.out_dir = out_dir

        self.observed_stats = {}
        self.simulations_dist_dict = {}
        self.obs_vs_sim_stats = None

    def init_observed_stats(self):
        self.observed_stats = read_json(self.observed_stats_json)

    def init_simulation_stats(self):
        graph_stats_jsons = find_files(self.simulations_dir, "graph_stats.json")
        if not graph_stats_jsons:
            raise FileNotFoundError(f"'graph_stats.json' files could not be retrieved from {self.simulations_dir}")
        
        #print(graph_stats_jsons)
        
        sim_dist_dict = {key : [] for key, value in self.observed_stats.items() if value is not None and key in self.ACCEPTED_KEYS}
        for json_file in graph_stats_jsons:
            sim_stats_dict = read_json(json_file)
            for key, value in sim_stats_dict.items():
                if key in sim_dist_dict and value is not None:
                    sim_dist_dict[key].append(value)
        self.simulations_dist_dict = sim_dist_dict
        return sim_dist_dict
    
    def process(self):
        self.init_observed_stats()
        self.init_simulation_stats()
        self.process_comparisons()
        self.process_plot_distributions()

    
    def process_comparisons(self):
        obs_vs_sim_stats = []
        
        with open(os.path.join(self.out_dir, "graph_stats_null_comparison_scores.json"), "w") as fh:
            for obs_null_stat in self.calc_process():
                fh.write(json.dumps(obs_null_stat))
                fh.write("\n")
                obs_vs_sim_stats.append(obs_null_stat)
        self.obs_vs_sim_stats = obs_vs_sim_stats


    def calc_process(self):
        for metric, dist_list in self.simulations_dist_dict.items():
            obs_val = self.observed_stats[metric]
            #print(type(obs_val))
            if isinstance(obs_val, list): continue
            z_score, mean_val, std_val = self.calc_z_score(obs_val, dist_list)
            p_val = self.calc_p(z_score) if z_score else None
            percentile_position = self.calc_percentile_position(obs_val, dist_list)

            obs_null_stat_dict = {
                "metric" : metric, 
                "obs_val" : obs_val, 
                "sim_mean":mean_val, 
                "sim_std":std_val,
                "z_score" : z_score,
                "p_val": p_val,
                "percentile_pos": percentile_position
                }
            
            yield obs_null_stat_dict
    
    @staticmethod
    def calc_z_score(observed_value, values_lst):
        # Calculate statistics
        mean_value = np.mean(values_lst)
        std_value = np.std(values_lst)
        z_score = (observed_value - mean_value) / std_value if std_value else None
        return z_score, mean_value, std_value
    
    @staticmethod
    def calc_p(z_score):##Two tailed p value calculation
        #p = (np.sum(np.abs(ensemble_entropies - mu) >= abs(obs_entropy - mu)) + 1) / (len(ensemble_entropies) + 1)
        return 2 * (1- stats.norm.cdf((z_score)))
    
    @staticmethod
    def calc_percentile_position(obs_val, dist_list):
        return (np.sum(np.array(dist_list) <= obs_val) / len(dist_list)) * 100
    

    def process_plot_distributions(self):
        from lib.paint_studio import NetworkStatsVisualizer
        # Create visualizer
        viz = NetworkStatsVisualizer(self.obs_vs_sim_stats, self.simulations_dist_dict)
        
        # Generate all plots
        print("Generating visualizations...")
        
        # Individual metric
        #fig1 = viz.plot_single_metric_distribution('density')
        #plt.savefig('01_single_metric.png', dpi=600, bbox_inches='tight')
        #plt.close()
        
        # Forest plot
        fig2 = viz.plot_z_score_forest()
        out_name = os.path.join(self.out_dir, '02_forest_plot.png')
        plt.savefig(out_name, dpi=600, bbox_inches='tight')
        plt.close()
        
        # P-value heatmap
        fig3 = viz.plot_p_value_heatmap()
        out_name = os.path.join(self.out_dir, '03_pvalue_heatmap.png')
        plt.savefig(out_name, dpi=600, bbox_inches='tight')
        plt.close()
        
        # Percentile plot
        fig4 = viz.plot_percentile_positions()
        out_name = os.path.join(self.out_dir, '04_percentile_plot.png')
        plt.savefig(out_name, dpi=600, bbox_inches='tight')
        plt.close()
        
        # Volcano plot
        fig5 = viz.plot_volcano()
        out_name = os.path.join(self.out_dir, '05_volcano_plot.png')
        plt.savefig(out_name, dpi=600, bbox_inches='tight')
        plt.close()
        
        # Multi-metric grid
        fig6 = viz.plot_multi_metric_grid(metrics_list=list(self.simulations_dist_dict.keys()))
        out_name = os.path.join(self.out_dir, '06_multi_metric_grid.png')
        plt.savefig(out_name, dpi=600, bbox_inches='tight')
        plt.close()
        
        # Q-Q plot
        #fig7 = viz.plot_qq_plot('density')
        #plt.savefig('07_qq_plot.png', dpi=600, bbox_inches='tight')
        #plt.close()
        
        # Summary report
        out_name = os.path.join(self.out_dir, '08_summary_report.png')
        fig8 = viz.create_summary_report(out_name)
        plt.close()
        
        print("All visualizations saved!")




class GDV_compare:

    def __init__(self, signature_file_1, signature_file_2, out_dir, name1="Network1", name2="Network2"):
        self.signature_file_1 = signature_file_1
        self.signature_file_2 = signature_file_2

        self.name1 = name1
        self.name2 = name2

        self.out_dir = out_dir

        self.similarity_score = None
        self.p_val = None


    def process(self, motif_number_up_to=43, genes_lst=None):
        df_ref = self.load_and_clean(self.signature_file_1, motif_number_upto=motif_number_up_to)
        df_comp = self.load_and_clean(self.signature_file_2, motif_number_upto=motif_number_up_to)

        if genes_lst:
            df_ref = df_ref.loc[genes_lst,:]
            df_comp = df_comp.loc[genes_lst,:]

        # 2. Global Similarity Analysis
        global_sim, p_value = self.calculate_global_similarity(df_ref, df_comp)
        self.similarity_score = global_sim
        self.p_val = p_value
        print(f"Global Graphlet Correlation Similarity: {global_sim:.4f}")
        print("(1.0 = Identical topology, 0.0 = Completely different)")
        print(f"p_value: {p_value}")

        write_to_json(
            {
                "network_1" : self.name1,
                "network_2": self.name2,
                "similarity_score" : self.similarity_score,
                "p_val" : self.p_val
            },
            os.path.join(self.out_dir, "gdv_network_similarity.json")
        )

        # 3. Gene Rewiring Analysis
        rewiring_scores = self.calculate_gene_rewiring(df_ref, df_comp)
        df_results = rewiring_scores.sort_values(ascending=False).to_frame()

        # Optional: Filter for Transcription Factors if you have a list
        # tf_list = ['HMGA2', 'TBX20', 'DTL', ...] 
        # top_tfs = df_results[df_results.index.isin(tf_list)]
        df_results.to_csv(os.path.join(self.out_dir, "gdv_rewire_scores.tsv"), sep="\t")

        print("\nTop 10 Most Rewired Genes:")
        print(df_results.head(10))

        # 4. Plotting
        from adjustText import adjust_text
        plt.figure(figsize=(12, 6))

        # Histogram with KDE
        sns.histplot(df_results['Rewiring_Score'], kde=True, bins=60, color='#2c3e50', alpha=0.6)

        # Get the y-axis limit for positioning
        ylim = plt.ylim()
        y_max = ylim[1]

        # Highlight top genes with adjustText
        top_genes = df_results.head(5)
        texts = []
        for idx, gene_score_tup in enumerate(top_genes['Rewiring_Score'].items()):
            gene = gene_score_tup[0]
            score = gene_score_tup[1]
            plt.axvline(score, color='violet', linestyle='--', alpha=0.5)
            
            # Create text annotation
            txt = plt.text(score, y_max * 0.7, f'{gene}', 
                        rotation=45, verticalalignment='bottom', 
                        color='red', fontsize=9)
            texts.append(txt)

        # Adjust text to avoid overlaps
        adjust_text(texts, 
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.5, alpha=0.7),
                    expand_points=(1.5, 1.5),
                    force_text=(0.5, 1.0))

        plt.title(f'Distribution of Gene Signature Changes\nGlobal Similarity: {global_sim:.2f}', fontsize=14)
        plt.xlabel('Euclidean Distance (Signature Rewiring)', fontsize=12)
        plt.ylabel('Number of Genes', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        plt.savefig(os.path.join(self.out_dir, 'gene_rewiring_distribution.png'))
        plt.close()

    @staticmethod
    def load_and_clean(file_path, motif_number_upto=24):
        """Loads TSV and filters for the 11 non-redundant orbits (graphlets up to 4 nodes)."""
        df = pd.read_csv(file_path, sep='\t', index_col=0)
        # The 11 non-redundant orbits for 2-4 node graphlets
        # Note: G3 is redundant (triangles can be inferred from G0, G1, G2)
        #keep_orbits = ['G0', 'G1', 'G2', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11']
        return df.iloc[:,:motif_number_upto]#[keep_orbits]

    @staticmethod
    def calculate_global_similarity(df1, df2, method="spearman"):
        """
        Calculates the Global Graphlet Correlation Similarity.
        Metric: Spearman Correlation between the flattened upper-triangles of the GCMs.
        Range: 1.0 (Identical topology) to 0.0 (Unrelated) to -1.0 (Opposite).
        """

        # 1. Compute Graphlet Correlation Matrix (GCM) for each network
        gcm1 = df1.corr(method=method)
        gcm2 = df2.corr(method=method)

        # 2. Extract upper triangle values (excluding diagonal)
        triu_indices = np.triu_indices_from(gcm1, k=1)
        vec1 = gcm1.values[triu_indices]
        vec2 = gcm2.values[triu_indices]

        # 3. Compute correlation between the two topology vectors
        sim_func = None
        if method == "spearman":
            sim_func = stats.spearmanr
        elif method == "pearson":
            sim_func = stats.pearsonr
        else:
            raise ValueError(f"{method} not implemented")

        similarity, p_val = sim_func(vec1, vec2)
        return similarity, p_val

    @staticmethod
    def calculate_gene_rewiring(df1, df2):
        """
        Calculates Euclidean distance between the signatures for common genes.
        """
        # Find common genes
        common_genes = df1.index.intersection(df2.index)
        print(f"Analyzing {len(common_genes)} common genes...")
        
        d1 = df1.loc[common_genes]
        d2 = df2.loc[common_genes]
        
        # Euclidean distance per row (gene)
        # dist = sqrt(sum((a-b)^2))
        diff = d1.values - d2.values
        distances = np.sqrt(np.sum(diff**2, axis=1))
        
        #distances = calculate_gdvd_log(d1.values, d2.values)

        ## Manhattan
        # Calculate Absolute Difference per orbit
        # Sum across columns (axis=1) to get distance per gene
        #abs_diff = np.abs(d1.values - d2.values)    
        #distances = np.sum(abs_diff, axis=1)
        
        return pd.Series(distances, index=common_genes, name='Rewiring_Score')

    @staticmethod
    def calculate_gdvd_log(u, v):
        """
        Computes the Log-based Graphlet Degree Vector Distance (GDVD).
        Robust to scale differences (hubs vs peripheral nodes).
        u, v: Arrays of 11 non-redundant orbit counts.
        """
        # Log transform (add 1 to avoid log(0))
        u_log = np.log(u + 1)
        v_log = np.log(v + 1)
        
        # Denominator scales by the magnitude of the larger value
        # (prevents small changes in large hubs from dominating)
        denom = np.log(np.maximum(u, v) + 2)
        
        # Average normalized difference across all 11 orbits
        term = np.abs(u_log - v_log) / denom
        #return term
        return np.mean(term)

class CentralityMetricsComparator:
    """Compare centrality metrics between two networks and identify significant changes."""
    
    CENTRALITY_METRICS = {
        'pagerank',
        'betweenness_centrality',
        'eigenvector_centrality',
        'degree_centrality',
        'in_degree_centrality',
        'out_degree_centrality',
        'closeness_centrality',
        'harmonic_centrality',
        'katz_centrality',
        'triangles',
        #'eccentricity',
    }
    
    def __init__(self, 
                 df1: pd.DataFrame,
                 df2: pd.DataFrame,
                 name1: str = "Network 1",
                 name2: str = "Network 2",
                 metrics: Optional[List[str]] = None,
                 node_column: Optional[str] = None):
        """
        Initialize with two networks or two DataFrames with pre-calculated centralities.
        
        Args:
            df1: DataFrame with pre-calculated centralities for network 1
            df2: DataFrame with pre-calculated centralities for network 2
            metrics: List of centrality metrics to compare. If None, uses all common columns
            node_column: Name of column containing node IDs. If None, uses index
        """
        #self.G1 = G1
        #self.G2 = G2
        self.node_column = node_column
        self.name1 = name1
        self.name2 = name2

        ## See: https://pmc.ncbi.nlm.nih.gov/articles/PMC12271745/
        self.jaccard_index = None
        
        # Determine if we're using DataFrames or computing from graphs
        if df1 is not None and df2 is not None:
            self._init_from_dataframes(df1, df2, metrics)
        #elif G1 is not None and G2 is not None:
        #    self._init_from_graphs(G1, G2, metrics)
        else:
            raise ValueError("Must provide df1, df2")
        
    
    def _init_from_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame, metrics: Optional[List[str]]):
        """Initialize from pre-calculated centrality DataFrames."""
        # Get node identifiers
        if self.node_column:
            nodes1 = set(df1[self.node_column])
            nodes2 = set(df2[self.node_column])
        else:
            nodes1 = set(df1.index)
            nodes2 = set(df2.index)
        
        self.common_nodes = nodes1 & nodes2

        if len(self.common_nodes) == 0:
            raise ValueError("No common nodes found between dataframes")
        
        # Calculate Jaccard index of nodes (size of intersection over size of union)
        union_nodes = nodes1 | nodes2
        self.jaccard_index = len(self.common_nodes) / len(union_nodes)

        # Determine which metrics to use
        exclude_cols = {self.node_column} if self.node_column else set()
        available_metrics1 = set(df1.columns) - exclude_cols
        available_metrics2 = set(df2.columns) - exclude_cols
        common_metrics = available_metrics1 & available_metrics2
        
        if metrics is None:
            self.metrics = sorted(self.CENTRALITY_METRICS.intersection(list(common_metrics)))
        else:
            self.metrics = [m for m in metrics if m in common_metrics]
            invalid = set(metrics) - common_metrics
            if invalid:
                warnings.warn(f"Metrics not found in both dataframes: {invalid}")
        
        if not self.metrics:
            raise ValueError("No valid metrics found in both dataframes")
        
        # Store centralities as dictionaries
        self.centralities1 = {}
        self.centralities2 = {}
        
        #print(f"Loading {len(self.metrics)} centrality metrics from dataframes...")
        for metric in self.metrics:
            print(f"  - {metric}")
            
            if self.node_column:
                self.centralities1[metric] = dict(zip(df1[self.node_column], df1[metric]))
                self.centralities2[metric] = dict(zip(df2[self.node_column], df2[metric]))
            else:
                self.centralities1[metric] = df1[metric].to_dict()
                self.centralities2[metric] = df2[metric].to_dict()
        
       # print("Done!\n")
        
        # Store dataframes for additional features if needed
        self.df1 = df1
        self.df2 = df2
    
    def compute_changes(self, metric: str = None) -> pd.DataFrame:
        """
        Compute various change metrics for all common nodes.
        
        Args:
            metric: Specific metric to analyze. If None, uses first metric.
        """
        if metric is None:
            metric = self.metrics[0]
        
        if metric not in self.metrics:
            raise ValueError(f"Metric '{metric}' not computed. Available: {self.metrics}")
        
        results = []
        
        cent1 = self.centralities1[metric]
        cent2 = self.centralities2[metric]
        
        # Convert to arrays for vectorized operations
        cent1_vals = np.array([cent1.get(node, 0) for node in self.common_nodes])
        cent2_vals = np.array([cent2.get(node, 0) for node in self.common_nodes])
        
        # Handle NaN values
        cent1_vals = np.nan_to_num(cent1_vals, nan=0.0)
        cent2_vals = np.nan_to_num(cent2_vals, nan=0.0)
        
        # Compute statistics for normalization
        mean1, std1 = cent1_vals.mean(), cent1_vals.std()
        mean2, std2 = cent2_vals.mean(), cent2_vals.std()
        
        # Pre-compute sorted lists for ranking (to avoid recomputing for each node)
        sorted_cent1 = sorted([v for v in cent1.values() if not pd.isna(v)], reverse=True)
        sorted_cent2 = sorted([v for v in cent2.values() if not pd.isna(v)], reverse=True)
        
        for node in self.common_nodes:
            cent1_val = cent1.get(node, 0)
            cent2_val = cent2.get(node, 0)
            
            # Handle NaN
            cent1_val = 0 if pd.isna(cent1_val) else cent1_val
            cent2_val = 0 if pd.isna(cent2_val) else cent2_val
            
            # Raw difference
            raw_diff = cent2_val - cent1_val
            
            # Percentage change (avoiding division by zero)
            if cent1_val != 0:
                pct_change = (cent2_val - cent1_val) / abs(cent1_val) * 100
            else:
                pct_change = np.inf if cent2_val != 0 else 0
            
            # Z-scores (standardized values)
            z1 = (cent1_val - mean1) / std1 if std1 > 0 else 0
            z2 = (cent2_val - mean2) / std2 if std2 > 0 else 0
            z_diff = z2 - z1
            
            # Rank in each network - FIXED VERSION
            try:
                rank1 = sorted_cent1.index(cent1_val) + 1
            except ValueError:
                # Value not in list - assign rank based on where it would fit
                rank1 = sum(1 for v in sorted_cent1 if v > cent1_val) + 1
            
            try:
                rank2 = sorted_cent2.index(cent2_val) + 1
            except ValueError:
                # Value not in list - assign rank based on where it would fit
                rank2 = sum(1 for v in sorted_cent2 if v > cent2_val) + 1
            
            rank_change = rank1 - rank2  # Positive means improved rank
            
            # Percentile in each network
            percentile1 = stats.percentileofscore(cent1_vals, cent1_val)
            percentile2 = stats.percentileofscore(cent2_vals, cent2_val)
            percentile_change = percentile2 - percentile1
            
            degree1 = self.df1.loc[node, "degree"] if "degree" in self.df1.columns else None
            degree2 = self.df2.loc[node, "degree"] if "degree" in self.df2.columns else None
            
            result = {
                'node': node,
                'metric': metric,
                'value1': cent1_val,
                'value2': cent2_val,
                'network1_name': self.name1,
                'network2_name': self.name2,
                'raw_diff': raw_diff,
                'pct_change': pct_change,
                'z1': z1,
                'z2': z2,
                'z_diff': z_diff,
                'rank1': rank1,
                'rank2': rank2,
                'rank_change': rank_change,
                'percentile1': percentile1,
                'percentile2': percentile2,
                'percentile_change': percentile_change,
            }
            
            if degree1 is not None:
                result['degree1'] = int(degree1)
                result['degree2'] = int(degree2)
                result['degree_change'] = int(degree2 - degree1)
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def compute_all_changes(self) -> pd.DataFrame:
        """Compute changes for all metrics and return combined DataFrame."""
        all_results = []
        
        for metric in self.metrics:
            df = self.compute_changes(metric)
            all_results.append(df)
        
        return pd.concat(all_results, ignore_index=True)
    
    def permutation_test(self, node: str, metric: str, n_permutations: int = 1000) -> float:
        """
        Perform permutation test to compute p-value for a node's centrality change.
        
        Args:
            node: Node to test
            metric: Centrality metric to test
            n_permutations: Number of random permutations
            
        Returns:
            p-value indicating significance of change
        """
        if node not in self.common_nodes:
            return np.nan
        
        if metric not in self.metrics:
            raise ValueError(f"Metric '{metric}' not computed")
        
        cent1 = self.centralities1[metric]
        cent2 = self.centralities2[metric]
        
        observed_diff = cent2.get(node, 0) - cent1.get(node, 0)
        
        # Get all centrality values (excluding NaN)
        all_cent1 = [v for v in cent1.values() if not pd.isna(v)]
        all_cent2 = [v for v in cent2.values() if not pd.isna(v)]
        
        if not all_cent1 or not all_cent2:
            return np.nan
        
        # Permutation test: randomly shuffle assignments
        null_distribution = []
        for _ in range(n_permutations):
            # Randomly sample from each distribution
            random_cent1 = np.random.choice(all_cent1)
            random_cent2 = np.random.choice(all_cent2)
            null_distribution.append(random_cent2 - random_cent1)
        
        null_distribution = np.array(null_distribution)
        
        # Two-tailed p-value
        p_value = (np.sum(np.abs(null_distribution) >= np.abs(observed_diff)) + 1) / (len(null_distribution) + 1)

        
        return p_value
    
    
    def find_significant_changes(self, metric: str = None, top_n: int = 10, 
                                 n_permutations: int = 1000,
                                 alpha: float = 0.05) -> pd.DataFrame:
        """
        Identify most significantly changed nodes with p-values.
        
        Args:
            metric: Centrality metric to analyze. If None, uses first metric.
            top_n: Number of top changed nodes to test
            n_permutations: Number of permutations for significance test
            alpha: Significance level
            
        Returns:
            DataFrame with significant changes and p-values
        """
        if metric is None:
            metric = self.metrics[0]
        
        df = self.compute_changes(metric)
        
        # Sort by absolute z-score difference (most significant changes)
        df['abs_z_diff'] = df['z_diff'].abs()
        df_sorted = df.sort_values('abs_z_diff', ascending=False).head(top_n)
        
        # Compute p-values for top candidates
        print(f"Computing p-values for top {top_n} nodes in '{metric}' (this may take a moment)...")
        p_values = []
        for node in df_sorted['node']:
            p_val = self.permutation_test(node, metric, n_permutations)
            p_values.append(p_val)
        
        df_sorted = df_sorted.copy()
        df_sorted['p_value'] = p_values
        df_sorted['significant'] = df_sorted['p_value'] < alpha
        
        # Bonferroni correction for multiple testing
        df_sorted['p_value_bonferroni'] = df_sorted['p_value'] * len(df_sorted)
        df_sorted['significant_bonferroni'] = df_sorted['p_value_bonferroni'] < alpha
        
        return df_sorted.sort_values('p_value')
    
    def find_all_significant_changes(self, top_n: int = 10, 
                                     n_permutations: int = 1000,
                                     alpha: float = 0.05) -> Dict[str, pd.DataFrame]:
        """Find significant changes for all metrics."""
        results = {}
        
        for metric in self.CENTRALITY_METRICS:
            if all(pd.isna(self.df1[metric])) or all(pd.isna(self.df2[metric])): continue
            print(f"\nAnalyzing {metric}...")
            results[metric] = self.find_significant_changes(
                metric=metric, 
                top_n=top_n, 
                n_permutations=n_permutations,
                alpha=alpha
            )
        
        return results
    
    def visualize_changes(self, df: pd.DataFrame, metric: str = None, top_n: int = 15, save=None):
        """Create visualizations of centrality changes."""
        if metric is None:
            metric = df['metric'].iloc[0]

         # Get network names from the dataframe
        name1 = df['network1_name'].iloc[0] if 'network1_name' in df.columns else "Network 1"
        name2 = df['network2_name'].iloc[0] if 'network2_name' in df.columns else "Network 2"
        
        df_plot = df.head(top_n).copy()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Centrality Changes: {metric}\n{name1} vs {name2}', 
                     fontsize=16, fontweight='bold')
        
        # 1. Raw centrality comparison
        ax1 = axes[0, 0]
        x = np.arange(len(df_plot))
        width = 0.35
        ax1.bar(x - width/2, df_plot['value1'], width, label=name1, alpha=0.8, color='steelblue')
        ax1.bar(x + width/2, df_plot['value2'], width, label=name2, alpha=0.8, color='coral')
        ax1.set_xlabel('Node', fontsize=11)
        ax1.set_ylabel(f'{metric}', fontsize=11)
        ax1.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df_plot['node'], rotation=45, ha='right', fontsize=9)
        ax1.legend(fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Z-score changes
        ax2 = axes[0, 1]
        colors = ['red' if x < 0 else 'green' for x in df_plot['z_diff']]
        ax2.barh(range(len(df_plot)), df_plot['z_diff'], color=colors, alpha=0.7)
        ax2.set_yticks(range(len(df_plot)))
        ax2.set_yticklabels(df_plot['node'], fontsize=9)
        ax2.set_xlabel('Z-score Change', fontsize=11)
        ax2.set_title(f'Standardized Change ({name1} → {name2})', fontsize=12, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax2.grid(axis='x', alpha=0.3)
        
        # Add text labels for direction
        ax2.text(0.98, 0.02, f'Higher in {name2} →', transform=ax2.transAxes,
                ha='right', va='bottom', fontsize=9, color='green', style='italic')
        ax2.text(0.02, 0.02, f'← Higher in {name1}', transform=ax2.transAxes,
                ha='left', va='bottom', fontsize=9, color='red', style='italic')
        
        # 3. Percentile changes
        ax3 = axes[1, 0]
        colors = ['red' if x < 0 else 'green' for x in df_plot['percentile_change']]
        ax3.barh(range(len(df_plot)), df_plot['percentile_change'], color=colors, alpha=0.7)
        ax3.set_yticks(range(len(df_plot)))
        ax3.set_yticklabels(df_plot['node'], fontsize=9)
        ax3.set_xlabel('Percentile Change', fontsize=11)
        ax3.set_title(f'Percentile Rank Change ({name1} → {name2})', fontsize=12, fontweight='bold')
        ax3.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. P-values
        ax4 = axes[1, 1]
        if 'p_value' in df_plot.columns:
            colors = ['green' if x < 0.05 else 'orange' if x < 0.1 else 'red' 
                     for x in df_plot['p_value']]
            ax4.barh(range(len(df_plot)), -np.log10(df_plot['p_value'].clip(lower=1e-10)), 
                    color=colors, alpha=0.7)
            ax4.set_yticks(range(len(df_plot)))
            ax4.set_yticklabels(df_plot['node'], fontsize=9)
            ax4.set_xlabel('-log10(p-value)', fontsize=11)
            ax4.set_title('Statistical Significance', fontsize=12, fontweight='bold')
            ax4.axvline(x=-np.log10(0.05), color='red', linestyle='--', 
                       linewidth=1.5, label='α = 0.05')
            ax4.axvline(x=-np.log10(0.01), color='darkred', linestyle='--', 
                       linewidth=1.5, label='α = 0.01')
            ax4.legend(fontsize=9)
            ax4.grid(axis='x', alpha=0.3)
        
        
        plt.tight_layout()
        if save:
            plt.savefig(save)
            plt.close()
        else:
            plt.show()
    
    def visualize_all_metrics(self, results_dict: Dict[str, pd.DataFrame], top_n: int = 10, save=None):
        """Create comparative visualization across all metrics."""
        n_metrics = len(results_dict)
        fig, axes = plt.subplots(n_metrics, 2, figsize=(14, 4*n_metrics))
        
        if n_metrics == 1:
            axes = axes.reshape(1, -1)
        
        # Get network names from first result
        first_df = list(results_dict.values())[0]
        name1 = first_df['network1_name'].iloc[0] if 'network1_name' in first_df.columns else "Network 1"
        name2 = first_df['network2_name'].iloc[0] if 'network2_name' in first_df.columns else "Network 2"
        
        fig.suptitle(f'Multi-Metric Comparison: {name1} vs {name2}', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        for idx, (metric, df) in enumerate(results_dict.items()):
            df_plot = df.head(top_n)
            
            # Z-score changes
            ax1 = axes[idx, 0]
            colors = ['red' if x < 0 else 'green' for x in df_plot['z_diff']]
            ax1.barh(range(len(df_plot)), df_plot['z_diff'], color=colors, alpha=0.7)
            ax1.set_yticks(range(len(df_plot)))
            ax1.set_yticklabels(df_plot['node'], fontsize=9)
            ax1.set_xlabel('Z-score Change', fontsize=10)
            ax1.set_title(f'{metric}: Standardized Change', fontsize=11, fontweight='bold')
            ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
            ax1.grid(axis='x', alpha=0.3)
            
            # Add direction labels
            ax1.text(0.98, 0.02, f'{name2} →', transform=ax1.transAxes,
                    ha='right', va='bottom', fontsize=8, color='green', style='italic')
            ax1.text(0.02, 0.02, f'← {name1}', transform=ax1.transAxes,
                    ha='left', va='bottom', fontsize=8, color='red', style='italic')
            
            # P-values
            ax2 = axes[idx, 1]
            if 'p_value' in df_plot.columns:
                colors = ['green' if x < 0.05 else 'orange' if x < 0.1 else 'red' 
                         for x in df_plot['p_value']]
                ax2.barh(range(len(df_plot)), -np.log10(df_plot['p_value'].clip(lower=1e-10)), 
                        color=colors, alpha=0.7)
                ax2.set_yticks(range(len(df_plot)))
                ax2.set_yticklabels(df_plot['node'], fontsize=9)
                ax2.set_xlabel('-log10(p-value)', fontsize=10)
                ax2.set_title(f'{metric}: Statistical Significance', fontsize=11, fontweight='bold')
                ax2.axvline(x=-np.log10(0.05), color='red', linestyle='--', 
                           linewidth=1.5, label='α = 0.05')
                if idx == 0:  # Only show legend on first plot
                    ax2.legend(fontsize=8)
                ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=600)
            plt.close()
        else:
            plt.show()
    
    def plot_centrality_heatmap(self, results_dict: Dict[str, pd.DataFrame], 
                           significance_level=0.05, filter_nonsignificant_nodes=True, 
                           figsize=None, save=None):
        """
        Create heatmap of z-score changes across all metrics and nodes.
        
        Parameters:
        -----------
        results_dict : Dict[str, pd.DataFrame]
            Dictionary with metric names as keys and comparison DataFrames as values
        significance_level : float
            P-value threshold for marking significance (default: 0.05)
        filter_nonsignificant_nodes : bool
            If True, exclude nodes that have no significant metrics (default: True)
        figsize : tuple
            Figure size (width, height). Auto-calculated if None
        save : str
            Path to save figure. If None, displays the plot
        """
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        # Combine all metrics into single dataframe
        all_data = []
        for metric, df in results_dict.items():
            for _, row in df.iterrows():
                all_data.append({
                    'node': row['node'],
                    'metric': metric,
                    'z_diff': row['z_diff'],
                    'p_value': row.get('p_value', None)  # Get p_value if it exists
                })
        
        df_combined = pd.DataFrame(all_data)
        if save:
            df_combined.to_csv(save.replace("png", "tsv"), sep="\t")
        
        # Pivot for heatmap: rows=metrics, columns=nodes, values=z_diff
        heatmap_data = df_combined.pivot(index='metric', columns='node', values='z_diff')
        
        # Create significance mask based on p-value
        p_value_data = df_combined.pivot(index='metric', columns='node', values='p_value')
        significance_mask = p_value_data < significance_level
        
        # Filter out nodes with no significant metrics
        if filter_nonsignificant_nodes:
            # Get nodes that have at least one significant metric
            nodes_with_significance = significance_mask.any(axis=0)
            significant_nodes = nodes_with_significance[nodes_with_significance].index.tolist()
            
            if len(significant_nodes) == 0:
                print(f"Warning: No nodes have significant metrics at p < {significance_level}")
                # Keep the most changed 10 nodes if none are significant
                if heatmap_data.shape[1] > 10:
                    # Compute node-wise max absolute z_diff and keep top 10 nodes
                    abs_change = heatmap_data.abs().max(axis=0)
                    most_changed_nodes = abs_change.sort_values(ascending=False).head(10).index.tolist()
                    heatmap_data = heatmap_data[most_changed_nodes]
                    p_value_data = p_value_data[most_changed_nodes]
                    significance_mask = significance_mask[most_changed_nodes]
                    print(f"Displaying the 10 most changed nodes out of {heatmap_data.shape[1]}")
            else:
                # Filter both heatmap data and p-value data
                heatmap_data = heatmap_data[significant_nodes]
                p_value_data = p_value_data[significant_nodes]
                significance_mask = significance_mask[significant_nodes]
                print(f"Displaying {len(significant_nodes)} nodes with at least one significant metric "
                    f"(filtered out {len(nodes_with_significance) - len(significant_nodes)} nodes)")
        
        # Auto-calculate figure size if not provided
        if figsize is None:
            n_nodes = len(heatmap_data.columns)
            n_metrics = len(heatmap_data.index)
            figsize = (max(12, n_nodes * 0.5), max(6, n_metrics * 0.6))
        
        # Get network names
        first_df = list(results_dict.values())[0]
        name1 = first_df['network1_name'].iloc[0] if 'network1_name' in first_df.columns else "Network 1"
        name2 = first_df['network2_name'].iloc[0] if 'network2_name' in first_df.columns else "Network 2"
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            heatmap_data,
            cmap='RdBu_r',  # Red for positive, Blue for negative
            center=0,
            annot=False,
            cbar_kws={'label': 'Z-score Change (z_diff)', 'shrink': 0.8},
            ax=ax,
            vmin=-abs(heatmap_data.values[~pd.isna(heatmap_data.values)]).max() if heatmap_data.notna().any().any() else -3,
            vmax=abs(heatmap_data.values[~pd.isna(heatmap_data.values)]).max() if heatmap_data.notna().any().any() else 3
        )
        
        # Add stars for significant changes (p < significance_level)
        for i, metric in enumerate(heatmap_data.index):
            for j, node in enumerate(heatmap_data.columns):
                if pd.notna(p_value_data.loc[metric, node]) and significance_mask.loc[metric, node]:
                    ax.text(j + 0.5, i + 0.5, '*', 
                        ha='center', va='center',
                        color='black', fontsize=16, fontweight='bold')
        
        # Add direction labels to colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.text(1.7, 0.99, f'{name2}', 
                    transform=cbar.ax.transAxes,
                    ha='left', va='top', fontsize=9, color='darkred', 
                    fontweight='bold', rotation=0)
        cbar.ax.text(1.7, 0.01, f'{name1}', 
                    transform=cbar.ax.transAxes,
                    ha='left', va='bottom', fontsize=9, color='darkblue', 
                    fontweight='bold', rotation=0)

        # Labels and title
        ax.set_xlabel('Nodes', fontsize=12, fontweight='bold')
        ax.set_ylabel('Centrality Metrics', fontsize=12, fontweight='bold')
        
        title_suffix = "\n(showing only nodes with significant changes)" if filter_nonsignificant_nodes else ""
        ax.set_title(f'Centrality Z-score Changes: {name1} vs {name2}\n(* = p < {significance_level}){title_suffix}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(save, dpi=800, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fig