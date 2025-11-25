from lib.grn_stats import GraphStats
from lib.utils import write_to_json, _make_serializable

import networkx as nx
from typing import List, Tuple, Dict, Optional, Set
import os

import pandas as pd
import time



##TODO: incorporate node_level output (top 10 in centrality, pagerank etc.)
##TODO: implement z-score calcuation for different conditions (condition_1 vs condition_2)
##TODO: basic plotting functions
##TODO implement clusters/communities with Pathways


class GRNArtist:
    """Main class for processing and analyzing Gene Regulatory Networks"""
    
    def __init__(self, tsv_input, output_dir, directed=True, n_cpu=None):
        self.tsv_input = tsv_input
        self.output_dir = output_dir
        self.directed = directed
        self.n_cpu = n_cpu


        self.graph: nx.Graph = None

        ## summary statistics
        self.graphstats_obj:GraphStats = None

        ###
        # calcuated here
        self.node_metrics_df:pd.DataFrame = None
        self.graph_stats:dict = None

        self.stats = None
    
    def initialise_graph(self):
        self.graph = self.read_from_tsv_edge_list(self.tsv_input, self.directed)
        
    
    def process_grn_statistics(self):
        ## Initialise data classes
        self.graphstats_obj = GraphStats(self.graph, self.directed)


        self.stats = {}
        self._calc_grn_stats() ## populates self.stats dict

        self.get_node_level_statistics()
        self.get_graph_level_statistics()



    def get_graph_level_statistics(self):
        stats = self.stats.copy()
        stats["n_communities"] = len(stats["leiden_communities"])

        for node_stat in [
            "betweenness_centrality", "eigenvector_centrality", "pagerank", "degree_centrality", 
            "in_degree_centrality", "out_degree_centrality", "harmonic_centrality", "triangles", 
            "closeness_centrality", "eccentricity", "leiden_communities"
            ]:
            stats.pop(node_stat)
        self.graph_stats = stats
        return self.graph_stats
        
    def get_node_level_statistics(self):
        centrality_stats_keys = [
           "betweenness_centrality", "eigenvector_centrality", "pagerank", "degree_centrality", 
            "in_degree_centrality", "out_degree_centrality", "harmonic_centrality", "triangles", 
            "closeness_centrality", "eccentricity", #"leiden_communities"
            ]
        
        node_metrics = {centrality_key:self.stats.get(centrality_key, {}) for centrality_key in centrality_stats_keys}
        #node_metrics.update(self.stats["betweenness_centrality"])
        #node_metrics.update(self.stats["eigenvector_centrality"])
        #node_metrics.update(self.stats["pagerank"])
        #node_metrics.update(self.stats["degree_centrality"])
        #node_metrics.update(self.stats["in_degree_centrality"])
        #node_metrics.update(self.stats["out_degree_centrality"])
        #node_metrics.update(self.stats["harmonic_centrality"])
        #node_metrics.update(self.stats["triangles"])
        #node_metrics.update(self.stats["closeness_centrality"])
        #node_metrics.update(self.stats["eccentricity"])
        node_metrics_df = pd.DataFrame(node_metrics)

        node_metrics_df["is_in_dominating_set"] = node_metrics_df.index.isin(self.stats["min_weighted_dominating_set"])
        
        leiden_communities = self.stats["leiden_communities"] #[{2, 3, 5, 7, 8}, {0, 1, 4, 6, 9}] ## community list of sets
        
        for community_idx, community_set in enumerate(leiden_communities):
            node_metrics_df.loc[community_set]["leiden_community"] = community_idx
        
        node_metrics_df["n_targets"] = node_metrics_df.index.apply(lambda x: self.graphstats_obj.query_n_descendants(x))


        self.node_metrics_df = node_metrics_df
        return self.node_metrics_df


    def _calc_grn_stats(self):
        ## Calculate eigenvector and laplacian matrices
        self.graphstats_obj.calc_adjacency_maxtix()
        self.graphstats_obj.calc_laplacian_matrix()
        self.graphstats_obj.calc_eigenvalues()

        ### Calculate centrality measures
        # Calculate and collect all graph-level statistics using methods from self.graphstats_obj
        gso = self.graphstats_obj

        

        # Define the functions and the corresponding keys
        parallel_calls = [
            ("betweenness_centrality", gso.get_betweenness_centrality),
            ("eigenvector_centrality", gso.get_eigenvector_centrality),
            ("pagerank", gso.get_pagerank),
            ("degree_centrality", gso.get_degree_centrality),
            ("in_degree_centrality", gso.get_in_degree_centrality),
            ("out_degree_centrality", gso.get_out_degree_centrality),
            ("harmonic_centrality", gso.get_harmonic_centrality),
            ("closeness_centrality", gso.get_closeness_centrality),
            ("triangles", gso.get_triangles),
            ("eccentricity", gso.get_eccentricity),
            ("leiden_communities", gso.get_leiden_communities),
            ("min_weighted_dominating_set", gso.get_min_weighted_dominating_set),
            ("n_edges", gso.get_n_edges),
            ("n_nodes", gso.get_n_nodes),
            ("avg_in_degree", gso.get_avg_in_degree),
            ("avg_out_degree", gso.get_avg_out_degree),
            ("avg_degree", gso.get_avg_degree),
            ("avg_closeness_centrality", gso.get_avg_closeness_centrality),
            ("avg_degree_centrality", gso.get_avg_degree_centrality),
            ("avg_betweenness_centrality", gso.get_avg_betweenness_centrality),
            ("avg_eigenvector_centrality", gso.get_avg_eigenvector_centrality),
            ("avg_pagerank_score", gso.get_avg_pagerank_score),
            ("avg_eccentricity", gso.get_avg_eccentricity),
            ("center", gso.get_center),
            ("diameter", gso.get_diameter),
            ("density", gso.get_density),
            ("transitivity", gso.get_transitivity),
            ("n_isolate_subgraphs", gso.get_n_isolate_subgraphs),
            ("n_triangles", gso.get_n_triangles),
            ("degree_assortativity", gso.get_degree_assortativity),
            ("degree_centralization", gso.get_degree_centralization),
            ("average_clustering_coeff", gso.get_average_clustering_coeff),
            ("large_clique_size", gso.get_large_clique_size),
            ("global_efficiency", gso.get_global_efficiency),
            ("local_efficiency", gso.get_local_efficiency),
            ("shannon_vertex_entropy", gso.get_shannon_vertex_entropy),
            ("structural_entropy", gso.get_structural_entropy),
            ("von_neumann_entropy", gso.get_von_neumann_entropy),
            ("shannon_degree_centrality_entropy", gso.get_shannon_degree_centrality_entropy),
            ("shannon_betweenness_centrality_entropy", gso.get_shannon_betweenness_centrality_entropy),
            ("shannon_pagerank_centrality_entropy", gso.get_shannon_pagerank_centrality_entropy),
            ("proxy_criticality_branching_ratio", gso.get_proxy_criticality_branching_ratio),
            ("proxy_average_sensitivity", gso.get_proxy_average_sensitivity),
            ("sigma", gso.get_sigma),
            ("omega", gso.get_omega)
        ]


        self.stats = {key:func_call() for key, func_call in parallel_calls}

        # Parallel execution of function-based metrics
        #import concurrent.futures
        #with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
        #    future_to_key = {executor.submit(func): key for key, func in parallel_calls}
        #    for future in concurrent.futures.as_completed(future_to_key):
        #        key = future_to_key[future]
        #        try:
        #            stats[key] = future.result()
        #        except Exception as exc:
        #            print(f"could not calculate for {key}\ngot error:\n{exc}")
        #            stats[key] = None  # or handle/log the exception as needed

        return self.stats
    

    def get_top_nodes(self, metric_column:str|List, top=20):
        return self.node_metrics_df[metric_column].nlargest(top)
        
    
    def write_node_stats(self):
        self.node_metrics_df.to_csv(os.path.join(self.output_dir, "node_stats.tsv"), index=True, sep='\t')
    
    def write_graph_stats(self):
        write_to_json(_make_serializable(self.graph_stats), os.path.join(self.output_dir, "graph_stats.json"))

    def plot_stats(self):
        import matplotlib.pyplot as plt
        ...
    
    def process_grn(self):
        self.initialise_graph()
        self.process_grn_statistics()
        self.write_graph_stats()
        self.write_node_stats()
        
    @staticmethod
    def read_from_tsv_edge_list(tsv_path, directed=True)->nx.Graph|nx.DiGraph:
        """
        Load GRN from tsv file
        """
        import pandas as pd

        grn_edgelist_df = pd.read_csv(tsv_path, sep="\t")

        col_names = grn_edgelist_df.columns

        if "source" not in col_names or "target" not in col_names:
            raise ValueError(f"{tsv_path} file does not contain 'source' or 'target' columns")
        

        return nx.from_pandas_edgelist(
                grn_edgelist_df, 
                source='source', 
                target='target', 
                edge_attr=True, 
                create_using=nx.DiGraph() if directed else nx.Graph(), 
                edge_key=None
                )



    
    

    