from lib.grn_stats import GraphLevelStats, CentralityStats, CriticallityStatistics, InformationExchangeStats
from lib.utils import _make_serializable

import networkx as nx
from typing import List, Tuple, Dict, Optional, Set
import os
import json
import inspect
import pandas as pd



##TODO: incorporate node_level output (top 10 in centrality, pagerank etc.)
##TODO: basic plotting functions


class GRNArtist:
    """Main class for processing and analyzing Gene Regulatory Networks"""
    
    def __init__(self, tsv_input, output_dir, directed=True):
        self.tsv_input = tsv_input
        self.output_dir = output_dir
        self.directed = directed


        #self.significance_threshold = significance_threshold
        #self.filter_threshold = filter_threshold

        self.graph: nx.Graph = None

        ## summary statistics
        # calculated in grn_stats.py
        self.graphlevelstats = None
        self.centralitystats = None
        self.entropystats = None
        self.criticallitystats = None
        self.informationexchangestats = None

        ###
        # calcuated here
        self.node_metrics_df:pd.DataFrame = None
        self.graph_stats:dict = None

        self.stats = None ## output dict to write
    
    def initialise_graph(self):
        self.graph = self.read_from_tsv_edge_list(self.tsv_input, self.directed)
    
    def process_grn_statistics(self):
        ## Initialise data classes
        self.graphlevelstats = GraphLevelStats(self.graph, self.directed)
        self.centralitystats = CentralityStats(self.graph, self.directed)
        self.criticallitystats = CriticallityStatistics(self.graph, self.directed)
        self.informationexchangestats = InformationExchangeStats(self.graph, self.directed)

        self.stats = {}
        
        print("calculating graph level statistics")
        self.stats.update(self.get_properties_dict(self.graphlevelstats))
        
        print("calculating centrality statistics")
        self.stats.update(self.get_properties_dict(self.centralitystats))
        
        print("calculating criticallity statistics")
        self.stats.update(self.get_properties_dict(self.criticallitystats))

        print("calculating information statistics")
        self.stats.update(self.get_properties_dict(self.informationexchangestats))


    def get_graph_level_statistics(self):
        stats = self.stats.copy()
        
        for node_stat in [
            "betweenness_centrality", "eigenvector_centrality", "pagerank", "degree_centrality", 
            "in_degree_centrality", "out_degree_centrality", "harmonic_centrality", "triangles", 
            "closeness_centrality", "eccentricity"
            ]:
            stats.pop(node_stat)



    def get_node_level_statistics(self):
        node_metrics = {}
        node_metrics.update(self.stats["betweenness_centrality"])
        node_metrics.update(self.stats["eigenvector_centrality"])
        node_metrics.update(self.stats["pagerank"])
        node_metrics.update(self.stats["degree_centrality"])
        node_metrics.update(self.stats["in_degree_centrality"])
        node_metrics.update(self.stats["out_degree_centrality"])
        node_metrics.update(self.stats["harmonic_centrality"])
        node_metrics.update(self.stats["triangles"])
        node_metrics.update(self.stats["closeness_centrality"])
        node_metrics.update(self.stats["eccentricity"])
        node_metrics_df = pd.DataFrame(node_metrics)

        node_metrics_df["is_in_dominating_set"] = node_metrics_df.index.isin(self.stats["min_weighted_dominating_set"])
        
        leiden_communities = [{2, 3, 5, 7, 8}, {0, 1, 4, 6, 9}] ## community list of sets
        
        for community_idx, community_set in enumerate(leiden_communities):
            node_metrics_df.loc[community_set]["leiden_community"] = community_idx
        
        node_metrics_df["n_targets"] = node_metrics_df.index.apply(lambda x: self.graphlevelstats.query_n_descendants(x))


        self.node_metrics_df = node_metrics_df
        return self.node_metrics_df

    

    def get_top_nodes(self, metric_column:str|List, top=20):
        return self.node_metrics_df[metric_column].nlargest(top)
        
    
    def write_stats_dict(self):
        ### control stats dict if contains non-serializable data
        serialized_stats_dict = _make_serializable(self.stats)
        ## write stats json
        with open(os.path.join(self.output_dir, "stats.json"), "w") as fh:
            json.dump(serialized_stats_dict, fh)


    def plot_stats(self):
        import matplotlib.pyplot as plt
        ...
        


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

    @staticmethod
    def get_properties_dict(obj):
        """
        Extract all properties (not attributes) from a dataclass instance
        and return them as a dictionary.
        """
        properties = {}
        
        # Get all members of the class
        for name, value in inspect.getmembers(type(obj)):
            # Check if it's a property descriptor
            if isinstance(value, property):
                # Get the property value from the instance
                try:
                    print(f"calculating for: {name}")
                    properties[name] = getattr(obj, name)
                except nx.NetworkXNotImplemented as e:
                    print(f"calculation for {name} is not implemented.\n{e}")
                    continue
        
        return properties

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

def comprehensive_node_analysis(self) -> Dict[str, Dict[str, float]]:
    """
    Calculate all key metrics for each node
    """
    deg_cent = self.degree_centrality()
    between_cent = self.betweenness_centrality()
    close_cent = self.closeness_centrality()
    harm_cent = self.harmonic_centrality()
    page_rank = self.pagerank()
    clustering = self.clustering_coefficient()
    
    analysis = {}
    for node in self.G.nodes():
        if self.directed:
            in_deg = self.G.in_degree(node)
            out_deg = self.G.out_degree(node)
        else:
            deg = self.G.degree(node)
            in_deg = out_deg = deg
        
        analysis[node] = {
            'in_degree': in_deg,
            'out_degree': out_deg,
            'degree_centrality': deg_cent[node],
            'betweenness_centrality': between_cent[node],
            'closeness_centrality': close_cent[node],
            'harmonic_centrality': harm_cent[node],
            'pagerank': page_rank[node],
            'clustering_coeff': clustering[node],
            'propagation_potential': self.perturbation_propagation_potential(node),
        }
    
    return analysis

def summary_report(self) -> Dict:
    """Generate a summary report of network statistics"""
    return {
        'Network Size': {
            'nodes': self.G.number_of_nodes(),
            'edges': self.G.number_of_edges(),
            'density': nx.density(self.G),
        },
        'Stability Metrics': self.network_stability(),
        'Clustering': {
            'transitivity': self.transitivity(),
            'average_clustering': self.average_clustering(),
        },
        'Top Hub Genes (by degree)': self._get_top_nodes(self.degree_centrality(), 5),
        'Top Influential (by PageRank)': self._get_top_nodes(self.pagerank(), 5),
        'Top Connectors (by betweenness)': self._get_top_nodes(self.betweenness_centrality(), 5),
    }

def _get_top_nodes(self, centrality_dict: Dict[str, float], n: int = 5) -> List[Tuple[str, float]]:
    """Get top n nodes by centrality"""
    return sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:n]




class ProcessGRNSummaryStatistics:
    """Summary statistics for Gene Regulatory Networks"""

    ### Initialisation method
    def set_directed_and_undirected_graphs(self):
        if self.directed:
            self.directed_graph = self.graph
            self.undirected_graph = self.graph.to_undirected()
        else:
            self.undirected_graph = self.graph
            self.directed_graph = None


    # ========== PATH QUERIES ==========
    
    def query_shortest_path(self, node1: str, node2: str) -> List[str]:
        """Get shortest path between two nodes"""
        return nx.shortest_path(self.graph, node1, node2)
    
    def path_exists(self, node1: str, node2: str) -> bool:
        """Check if path exists between two nodes"""
        return nx.has_path(self.graph, node1, node2)
    
    # ========== UTILITY METHODS ==========
    
    def get_eulerian_path(self) -> List[Tuple[str, str]]:
        """Get Eulerian path if it exists"""
        if self.has_eulerian_path:
            return list(nx.eulerian_path(self.graph))
        return []
    
    def get_significant_edges(self, significance_threshold: float) -> list[tuple[str, str]]:
        return [edge for edge in self.graph.edges(data=True) if edge[2]['weight'] > significance_threshold]
    
    def get_hubs(self, n_hubs: int) -> List[Tuple[str, int]]:
        """Get top n hubs by degree"""
        degrees = dict(self.graph.degree())
        return sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:n_hubs]
    
    def get_isolated_nodes(self) -> List[str]:
        """Get nodes with no connections"""
        return [node for node in self.graph.nodes() if self.graph.degree(node) == 0]
    
    def get_connected_components(self) -> List[set[str]]:
        """Get all connected components"""
        if self.directed:
            return list(nx.weakly_connected_components(self.graph))
        return list(nx.connected_components(self.graph))
    
    def get_cycles(self) -> List[set[str]]:
        """Get cycle basis (for undirected graphs)"""
        if self.directed:
            return [set(cycle) for cycle in nx.simple_cycles(self.graph)]
        return list(nx.cycle_basis(self.graph))
    
    

    