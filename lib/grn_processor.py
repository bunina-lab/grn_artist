from lib.grn_stats import GraphStats
from lib.utils import write_to_json, _make_serializable, call_subprocess
from lib.database_enrichments import process_database_enrichment
import tempfile

import networkx as nx
from typing import List, Literal, Tuple, Dict, Optional, Set, Union
import os

import pandas as pd
import numpy as np
import time



##TODO: incorporate node_level output (top 10 in centrality, pagerank etc.)
##TODO: implement z-score calcuation for different conditions (condition_1 vs condition_2)
##TODO: basic plotting functions
##TODO implement clusters/communities with Pathways


class GRNArtist:
    """Main class for processing and analyzing Gene Regulatory Networks"""
    
    def __init__(self, tsv_input, output_dir, directed=True, n_cpu=None, leiden_resolution=1.0):
        self.tsv_input = tsv_input
        self.output_dir = output_dir
        self.directed = directed
        self.n_cpu = n_cpu
        self.leiden_resolution = leiden_resolution


        self.graph: nx.Graph = None
        self.TF_list = None

        ## summary statistics
        self.graphstats_obj:GraphStats = None

        ###
        # calcuated here
        self.node_metrics_df:pd.DataFrame = None
        self.graph_stats:dict = None

        self.stats = None
    
    def process_grn(self):
        self.initialise_graph()
        self.process_grn_statistics()
        self.write_graph_stats()
        self.write_node_stats()
        self.plot_node_stats()
        print("Processing Graphlet Degree Vector")
        self.process_graphlet_degree_vector()
        print("Database Enrichment")
        process_database_enrichment(
            stats_df=self.node_metrics_df,
            graph=self.graph,
            outdir=self.output_dir
        )
        print("Analysis happily finished!")

    def initialise_graph(self):
        self.graph, self.TF_list = self.read_from_tsv_edge_list(self.tsv_input, self.directed)
        self.stats = {}
    
    def process_grn_statistics(self):
        ## Initialise data class
        self.graphstats_obj = GraphStats(
            graph=self.graph, 
            directed=self.directed,
            seed_nodes=set(self.TF_list)
            )

        self._calc_grn_stats() ## populates self.stats dict

        self.get_node_level_statistics()
        self.get_graph_level_statistics()



    def get_graph_level_statistics(self):
        stats = self.stats.copy()
        stats["n_communities"] = len(stats["leiden_communities"])

        for node_stat in [
            "betweenness_centrality", "eigenvector_centrality", "pagerank", "degree_centrality", 
            "in_degree_centrality", "out_degree_centrality", "harmonic_centrality", "triangles", 
            "closeness_centrality", "eccentricity", "leiden_communities", "min_weighted_dominating_set",
            "katz_centrality"
            ]:
            stats.pop(node_stat)
        self.graph_stats = stats
        return self.graph_stats
        
    def get_node_level_statistics(self):
        centrality_stats_keys = [
           "betweenness_centrality", "eigenvector_centrality", "pagerank", "degree_centrality", 
            "in_degree_centrality", "out_degree_centrality", "harmonic_centrality", "triangles", 
            "closeness_centrality", "eccentricity", "katz_centrality" #"leiden_communities"
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
            node_metrics_df.loc[list(community_set), "leiden_community"] = community_idx
        
        #node_metrics_df["n_targets"] = node_metrics_df.index.to_series().apply(lambda x: self.graphstats_obj.query_n_descendants(x))


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
            ("katz_centrality", gso.get_katz_centrality),
            ("triangles", gso.get_triangles),
            ("eccentricity", gso.get_eccentricity),
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
            #("large_clique_size", gso.get_large_clique_size),
            ("global_efficiency", gso.get_global_efficiency),
            #("local_efficiency", gso.get_local_efficiency),
            ("shannon_vertex_entropy", gso.get_shannon_vertex_entropy),
            ("structural_entropy", gso.get_structural_entropy),
            ("von_neumann_entropy", gso.get_von_neumann_entropy),
            ("shannon_degree_centrality_entropy", gso.get_shannon_degree_centrality_entropy),
            ("shannon_betweenness_centrality_entropy", gso.get_shannon_betweenness_centrality_entropy),
            ("shannon_pagerank_centrality_entropy", gso.get_shannon_pagerank_centrality_entropy),
            ("proxy_criticality_branching_ratio", gso.get_proxy_criticality_branching_ratio),
            ("proxy_average_sensitivity", gso.get_proxy_average_sensitivity),
            #("sigma", gso.get_sigma),
            #("omega", gso.get_omega)
        ]


        #self.stats = {key:func_call() for key, func_call in parallel_calls}

        for key, func_call in parallel_calls:
            print(f"Calculating for {key}\n")
            start = time.time()
            self.stats.update({key:func_call()})
            end = time.time()
            print(f"Calculated in {end-start} seconds\n")
        
        print("Getting leiden communities")
        self.stats.update({"leiden_communities": gso.get_leiden_communities(resolution=self.leiden_resolution)}),


        self.stats.update({
            "undireced_n_egdes" : gso.undireced_n_egdes,
            "n_self_loops" : gso.n_self_loops
            })

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
    

    def get_top_nodes(self, metric_column:Union[str,List], top=20):
        return self.node_metrics_df[metric_column].nlargest(top)
        
    
    def write_node_stats(self):
        self.node_metrics_df.to_csv(os.path.join(self.output_dir, "node_stats.tsv"), index=True, sep='\t')
    
    def write_graph_stats(self):
        write_to_json(_make_serializable(self.graph_stats), os.path.join(self.output_dir, "graph_stats.json"))

    def plot_node_stats(self):
        from lib.paint_studio import plot_multiple_metrics
        plot_multiple_metrics(
            self.node_metrics_df, 
            ["betweenness_centrality", "eigenvector_centrality", "pagerank", "degree_centrality", "harmonic_centrality", "closeness_centrality", "katz_centrality"], 
            top_n=10, 
            ncols=2, 
            figsize=(18, 8),
            out_path=os.path.join(self.output_dir, "node_stats_plot.png")
            )
    
        
    @staticmethod
    def read_from_tsv_edge_list(tsv_path, directed=True)->Union[nx.Graph,nx.DiGraph]:
        """
        Load GRN from tsv file
        """
        import pandas as pd

        grn_edgelist_df = pd.read_csv(tsv_path, sep="\t")

        col_names = grn_edgelist_df.columns

        if "source" not in col_names or "target" not in col_names:
            raise ValueError(f"{tsv_path} file does not contain 'source' or 'target' columns")
        

        return (nx.from_pandas_edgelist(
                grn_edgelist_df, 
                source='source', 
                target='target', 
                edge_attr=True, 
                create_using=nx.DiGraph() if directed else nx.Graph(), 
                edge_key=None
                ), 
                list(grn_edgelist_df["source"].unique()),
        )


    def process_graphlet_degree_vector(self, similarity_metric:Literal["cosine", "euclidean"]="cosine"):
        from lib.graphlet_analysis import GraphletAnalyzer

        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering

        out_dir = os.path.join(self.output_dir, "graphlet_analysis")
        gdv_matrix, node2id_map, id2node_map = self.run_orca(outdir=out_dir)

        graphlet_analyser = GraphletAnalyzer(gdv_matrix, id2node_map, out_dir=out_dir)
        graphlet_analyser.initialise_analyser()
        signature_matrix = graphlet_analyser.get_signatures() ## n_nodes X n_orbits
        similarity_matrix = graphlet_analyser.compute_similarity(metric=similarity_metric) ## n_nodes X n_nodes

        # Save signature matrix (graphlet signatures) as TSV
        # signature_matrix: numpy array, node2id_map: node label -> index, id2node_map: index -> node label
        signature_df = pd.DataFrame(
            signature_matrix,
            index=[id2node_map[i] for i in range(signature_matrix.shape[0])],
            columns=[f"G{i}" for i in range(signature_matrix.shape[1])]
        )
        signature_df.index.name = "node"
        # The columns are orbits, name as G0, G1, ..., if available from graphlet_analyser
        if hasattr(graphlet_analyser, "graphlet_names") and graphlet_analyser.graphlet_names is not None:
            signature_df.columns = graphlet_analyser.graphlet_names
        else:
            signature_df.columns = [f"G{i}" for i in range(signature_matrix.shape[1])]
        signature_df.to_csv(os.path.join(out_dir, "graphlet_signatures.tsv"), sep="\t")

        # Save similarity matrix as TSV
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=[id2node_map[i] for i in range(similarity_matrix.shape[0])],
            columns=[id2node_map[i] for i in range(similarity_matrix.shape[1])],
        )
        similarity_df.index.name = "node"
        similarity_df.to_csv(os.path.join(out_dir, "similarity_matrix.tsv"), sep="\t")

        ## what is the similarity between TFs ?
        # Prepare similarity matrix among TFs
        tf_indices = list(map(lambda x: node2id_map[x], self.TF_list))
        tf_similarity_matrix = graphlet_analyser.compare_similarity(tf_indices)

        # Group TFs by similarity and plot their graphlet signatures

        # Use Agglomerative Clustering to bin TFs by similarity
        # You can adjust n_clusters or use a distance threshold as desired
        # Group TFs by their leiden cluster assignment in the main network
        # Ensure TFs are present in the node2id mapping
        leiden_communities = self.stats.get("leiden_communities", [])
        # Build node to leiden cluster lookup (node name -> cluster id)
        node_to_leiden = {}
        for cluster_id, node_set in enumerate(leiden_communities):
            for node in node_set:
                node_to_leiden[node] = cluster_id

        # Map TFs to their leiden cluster
        tf_to_leiden_cluster = {tf: node_to_leiden.get(tf, None) for tf in self.TF_list if tf in node2id_map}

        # Group TFs by leiden cluster
        from collections import defaultdict
        cluster_to_tfs = defaultdict(list)
        for tf, cluster_id in tf_to_leiden_cluster.items():
            if cluster_id is not None:
                cluster_to_tfs[cluster_id].append(tf)

        # For further downstream code using tf_cluster_labels, mimic that value for consistency:
        # tf_cluster_labels: contains the leiden cluster id per tf_indices order
        #idx_to_tf = {i: tf for i, tf in enumerate(self.TF_list) if tf in node2id_map}
        #tf_cluster_labels = [tf_to_leiden_cluster[idx_to_tf[i]] for i in range(len(idx_to_tf))]

        # Plot signature comparison for each cluster
        for cluster_id, tf_list in cluster_to_tfs.items():
            if not tf_list:
                continue
            tf_idx_in_matrix = [node2id_map[tf] for tf in tf_list]
            graphlet_analyser.plot_signature_comparison(tf_idx_in_matrix, use_names=True, save_plot_name=f"Graphlet Signatures: TF Cluster {cluster_id} ({len(tf_list)} TFs)")
            # GraphletAnalyzer's plot methods already handle saving or showing
        graphlet_analyser.plot_cluster_overlay(save_plot_name="cluster_overlay.png")
        ## save signature and similarity dataframes


    def run_orca(self, count_on:Literal["node", "edge"]="node", graphlet_size=5, outdir= None):
        ## https://academic.oup.com/bioinformatics/article/23/2/e177/202080
        # orca_path: path to ORCA binary (compiled from https://github.com/thocevar/orca)

        if graphlet_size == 5: expected_vector_size = 73 if count_on == "node" else 68
        elif graphlet_size == 4: 
            if count_on == "node":
                expected_vector_size = 14  
            else: 
                raise Exception("Graphlet size with 'edge' is not supported!")
        else: raise Exception("Graphlet size not supported!")

        tmp_folder = os.path.join(self.output_dir, "orca_process_tmp") if outdir is None else outdir
        os.makedirs(tmp_folder, exist_ok=True)

        in_fn = os.path.join(tmp_folder, "graph.txt")
        out_fn = os.path.join(tmp_folder, "gdv_counts.txt")
        #mapping = self._write_edgelist_for_orca(self.graph, in_fn)
        node2id_mapping = self._write_edgelist_for_orca(self.graph, in_fn)
        # ORCA modes: "node" for node-orbits (4- and 5-node), "edge" for edges, etc.
        # Here we ask ORCA to produce "node" counts for 4- and 5-node orbits (73 orbits).
        print("running orca")
        call_subprocess("bin/orca", [count_on, str(graphlet_size), in_fn, out_fn])
        # parse out_fn
        # ORCA outputs rows for each node, columns are orbit counts; for 5-node it returns 73 cols (or fewer if configured)

        ### Read output matrix
        mat = []
        with open(out_fn) as f:
            for line in f:
                row = list(map(int, line.strip().split()))
                if len(row) != expected_vector_size:
                    raise Exception(f"Unexpected column size in {out_fn}\nLen row: {len(row)}")
                mat.append(row)
        arr = np.array(mat, dtype=np.int64)  # shape (n, #orbits)
        assert arr.shape == (len(node2id_mapping), expected_vector_size); "Matrix size does not match"
        # reverse mapping: index->original node
        id2node_map = {idx:node for node,idx in node2id_mapping.items()}
        return arr, node2id_mapping, id2node_map
    
    
    @staticmethod
    def _write_edgelist_for_orca(G, file_path):
        # ORCA expects: first line "n m" then m lines "u v" with nodes numbered 0..n-1
        # Ensure nodes are integers 0..n-1 in order
        # filters self loop edges and multiple edges
        # returns mapping dict ('node' : 'id')
        mapping = {n:i for i,n in enumerate(sorted(G.nodes()))}
        n = len(mapping)
        edges = []
        for u,v in G.edges():
            a,b = mapping[u], mapping[v]
            if a==b: 
                continue
            if a>b: a,b=b,a
            edges.append((a,b))
        edges = sorted(set(edges))
        with open(file_path, "w") as f:
            f.write(f"{n} {len(edges)}\n")
            for a,b in edges:
                f.write(f"{a} {b}\n")
        return mapping

    

    