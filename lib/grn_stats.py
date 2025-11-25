import networkx as nx
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set


"""
Some sources that explain network statistics

https://ieeexplore.ieee.org/document/7456290

https://homes.cs.washington.edu/~anuprao/pubs/CSE533Autumn2010/lecture4.pdf

Topological benchmarking of algorithms to infer gene regulatory networks from single-cell RNA-seq data:
https://academic.oup.com/bioinformatics/article/40/5/btae267/7646844

https://arxiv.org/pdf/1805.01447
"""


@dataclass
class GraphStats:
    graph: nx.Graph | nx.DiGraph
    directed: bool

    adjacency_matrix = None
    laplacian_matrix = None
    eigenvalues = None
    
    # Graph Level Statistics
    avg_in_degree: Optional[float] = None
    avg_out_degree: Optional[float] = None
    avg_degree: Optional[float] = None
    n_edges: Optional[int] = None
    n_nodes: Optional[int] = None
    girth: Optional[int] = None
    density: Optional[float] = None
    is_connected: Optional[bool] = None
    is_DAG: Optional[bool] = None
    is_eulerian: Optional[bool] = None
    has_eulerian_path: Optional[bool] = None
    transitivity: Optional[float] = None
    n_isolate_subgraphs: Optional[int] = None
    triangles: Optional[Dict[str, int]] = None
    n_triangles: Optional[int] = None

    
    # Centrality statistics
    degree_centrality: Optional[Dict[str, float]] = None
    in_degree_centrality: Optional[Dict[str, float]] = None
    out_degree_centrality: Optional[Dict[str, float]] = None
    betweenness_centrality: Optional[Dict[str, float]] = None
    pagerank_centrality: Optional[Dict[str, float]] = None
    closeness_centrality: Optional[Dict[str, float]] = None
    eigenvector_centrality: Optional[Dict[str, float]] = None
    eccentricity: Optional[Dict[str, float]] = None
    katz_centrality: Optional[Dict[str, float]] = None
    harmonic_centrality: Optional[Dict[str, float]] = None
    pagerank: Optional[Dict[str, float]] = None
    center: Optional[List[str]] = None
    diameter: Optional[float] = None
    min_weighted_dominating_set: Optional[Set[str]] = None

    ## Avg calculations
    avg_closeness_centrality: Optional[float] = None
    avg_degree_centrality: Optional[float] = None
    avg_betweenness_centrality: Optional[float] = None
    avg_eigenvector_centrality: Optional[float] = None
    avg_pagerank_score: Optional[float] = None
    avg_eccentricity: Optional[float] = None
    
    ## Hub topology measures
    degree_assortativity: Optional[float] = None
    degree_centralization: Optional[float] = None
    
    ## Clustering
    average_clustering_coeff: Optional[float] = None
    large_clique_size: Optional[int] = None
    leiden_communities: Optional[List[Set]] = None
    
    ## Entropy
    shannon_vertex_entropy: Optional[float] = None
    structural_entropy: Optional[float] = None
    von_neumann_entropy: Optional[float] = None
    shannon_degree_centrality_entropy: Optional[float] = None
    shannon_betweenness_centrality_entropy: Optional[float] = None
    shannon_pagerank_centrality_entropy: Optional[float] = None
    shannon_closeness_centrality_entropy: Optional[float] = None

    ## Information exchange
    global_efficiency: Optional[float] = None
    local_efficiency: Optional[float] = None
    ### Small worldness
    omega: Optional[float] = None
    sigma: Optional[float] = None

    ## Criticallity stats
    proxy_criticality_branching_ratio: Optional[float] = None
    proxy_average_sensitivity: Optional[float] = None


    def calc_laplacian_matrix(self):
        if self.laplacian_matrix is None:
            self.laplacian_matrix = nx.directed_laplacian_matrix(self.graph, weight="weight") \
            if self.graph.is_directed \
            else nx.normalized_laplacian_matrix(self.graph, weight="weight")
        return self.laplacian_matrix
    
    def calc_eigenvalues(self, filter_value=1e-10):
        eigenvalues = np.linalg.eigvalsh(self.calc_laplacian_matrix())        
        # Filter positive eigenvalues (numerical stability)
        self.eigenvalues = eigenvalues[eigenvalues > filter_value]
        return self.eigenvalues
    
    def calc_adjacency_maxtix(self):
        if self.adjacency_matrix is None:
            self.adjacency_matrix = nx.adjacency_spectrum(self.graph)
        return self.adjacency_matrix
    
    def get_avg_in_degree(self) -> float:
        """Calculate average in-degree (for directed graphs)"""
        if self.avg_in_degree is None:
            if self.directed:
                self.avg_in_degree = float(np.mean([self.graph.in_degree(n) for n in self.graph.nodes()])) if self.graph.number_of_nodes() > 0 else 0.0
            else:
                self.avg_in_degree = 0.0
        return self.avg_in_degree

    def get_avg_out_degree(self) -> float:
        """Calculate average out-degree (for directed graphs)"""
        if self.avg_out_degree is None:
            if self.directed:
                self.avg_out_degree = float(np.mean([self.graph.out_degree(n) for n in self.graph.nodes()])) if self.graph.number_of_nodes() > 0 else 0.0
            else:
                self.avg_out_degree = 0.0
        return self.avg_out_degree

    def get_avg_degree(self) -> float:
        """Calculate average degree (for undirected graphs)"""
        if self.avg_degree is None:
            if not self.directed:
                self.avg_degree = float(np.mean([self.graph.degree(n) for n in self.graph.nodes()])) if self.graph.number_of_nodes() > 0 else 0.0
            else:
                # For directed, can take the mean of in-degree and out-degree if desired
                self.avg_degree = float(np.mean([self.graph.in_degree(n) + self.graph.out_degree(n) for n in self.graph.nodes()])) / 2 if self.directed and self.graph.number_of_nodes() > 0 else 0.0
        return self.avg_degree

    def get_n_edges(self) -> int:
        if self.n_edges is None:
            self.n_edges = self.graph.number_of_edges()
        return self.n_edges
    
    def get_n_nodes(self) -> int:
        if self.n_nodes is None:
            self.n_nodes = self.graph.number_of_nodes()
        return self.n_nodes

    def get_girth(self) -> int:
        """
        Girth is the shortest cycle of the graph
        """
        if self.girth is None:
            g = self.graph.to_undirected() if self.directed else self.graph
            self.girth = nx.girth(g)
        return self.girth
    
    def get_density(self) -> float:
        if self.density is None:
            self.density = float(nx.density(self.graph))
        return self.density

    #def get_n_cycles(self)->int:
    #    return len(nx.simple_cycles(self.graph))

    def get_is_connected(self) -> bool:
        """Check if graph is connected"""
        if self.is_connected is None:
            graph = self.graph if not self.directed else self.graph.to_undirected()
            self.is_connected = nx.is_connected(graph)
        return self.is_connected
    
    def get_is_DAG(self) -> bool:
        """Check if graph is a Directed Acyclic Graph"""
        if self.is_DAG is None:
            self.is_DAG = nx.is_directed_acyclic_graph(self.graph) if self.directed else False
        return self.is_DAG
    
    #def get_is_cyclic(self) -> bool:
    #    """Check if graph contains cycles"""
    #    return not self.is_DAG if self.directed else len(list(nx.simple_cycles(self.graph))) > 0

    def get_is_eulerian(self) -> bool:
        """Check if graph is Eulerian"""
        if self.is_eulerian is None:
            self.is_eulerian = nx.is_eulerian(self.graph)
        return self.is_eulerian
    
    def get_has_eulerian_path(self) -> bool:
        """Check if graph has an Eulerian path"""
        if self.has_eulerian_path is None:
            self.has_eulerian_path = nx.has_eulerian_path(self.graph)
        return self.has_eulerian_path
    
    def get_transitivity(self) -> float:
        """
        Global clustering coefficient: probability that adjacent nodes of a node
        are connected to each other (forms triangles).
        
        For directed graphs: counts directed triangles
        Range: 0-1
        
        Relevant for GRNs: high transitivity suggests feedback loops or
        tightly regulated modules. Lower in biological networks (usually 0.05-0.3)
        The transitivity is: T = 3x #triangles / #triads
        """
        if self.transitivity is None:
            self.transitivity = nx.transitivity(self.graph)
        return self.transitivity
    
    def get_n_isolate_subgraphs(self) -> int:
        if self.n_isolate_subgraphs is None:
            self.n_isolate_subgraphs = nx.algorithms.number_of_isolates(self.graph)
        return self.n_isolate_subgraphs
    
    def get_triangles(self) -> Dict[str,int]:
        "returns number of triangles per node"
        if self.triangles is None:
            g = self.graph.to_undirected() if self.directed else self.graph
            self.triangles = nx.triangles(g)
        return self.triangles
    
    def get_n_triangles(self) -> int:
        if self.n_triangles is None:
            triangles = self.get_triangles()
            self.n_triangles = sum(triangles.values())
        return self.n_triangles
    
   
    def get_sigma(self) -> float:
        "sigma coefficient of small world-ness"
        "TAKES TOO MUCH TIME"
        if self.sigma is None:
            g = self.graph.to_undirected() if self.directed else self.graph
            self.sigma = nx.sigma(g)
        return self.sigma
    
   
    def get_omega(self) -> float:
       "omega coefficient of small world-ness"
       "TAKES TOO MUCH TIME"
       if self.omega is None:
           g = self.graph.to_undirected() if self.directed else self.graph
           self.omega = nx.omega(g)
       return self.omega
    
    # ========== SENSITIVITY-RELATED MEASURES ==========
    def query_descendants(self, source_node) -> Set:
        return nx.descendants(self.graph, source_node) | {source_node}
    
    def query_n_descendants(self, source_node) -> int:
        return len(nx.descendants(self.graph, source_node) | {source_node})


    def perturbation_propagation_potential(self, source_node) -> float:
        """
        How far can a perturbation (knockout/knockdown) propagate from a single node?
        Relevant for GRNs: identifies which genes, if perturbed, affect many others
        """
        # BFS to find all reachable nodes
        reachable = self.query_descendants(source_node)
        return len(reachable) / self.graph.number_of_nodes()

    # ========== CENTRALITY MEASURES ==========
    def get_avg_closeness_centrality(self) -> Optional[float]:
        if self.avg_closeness_centrality is None:
            closeness_centrality = self.get_closeness_centrality()
            if closeness_centrality is None:
                self.avg_closeness_centrality = None
            else:
                vals = closeness_centrality.values()
                self.avg_closeness_centrality = sum(vals) / len(vals) if vals else None
        return self.avg_closeness_centrality

    def get_avg_degree_centrality(self) -> Optional[float]:
        """Average degree centrality of the graph"""
        if self.avg_degree_centrality is None:
            degree_centrality = self.get_degree_centrality()
            if degree_centrality is None:
                self.avg_degree_centrality = None
            else:
                vals = degree_centrality.values()
                self.avg_degree_centrality = sum(vals) / len(vals) if vals else None
        return self.avg_degree_centrality

    def get_avg_betweenness_centrality(self) -> Optional[float]:
        """Average betweenness centrality of the graph"""
        if self.avg_betweenness_centrality is None:
            betweenness_centrality = self.get_betweenness_centrality()
            if betweenness_centrality is None:
                self.avg_betweenness_centrality = None
            else:
                vals = betweenness_centrality.values()
                self.avg_betweenness_centrality = sum(vals) / len(vals) if vals else None
        return self.avg_betweenness_centrality

    def get_avg_eigenvector_centrality(self) -> Optional[float]:
        """Average eigenvector centrality of the graph"""
        if self.avg_eigenvector_centrality is None:
            eigenvector_centrality = self.get_eigenvector_centrality()
            if eigenvector_centrality is None:
                self.avg_eigenvector_centrality = None
            else:
                vals = eigenvector_centrality.values()
                self.avg_eigenvector_centrality = sum(vals) / len(vals) if vals else None
        return self.avg_eigenvector_centrality

    def get_avg_pagerank_score(self) -> Optional[float]:
        """Average PageRank score of the graph"""
        if self.avg_pagerank_score is None:
            pagerank = self.get_pagerank()
            if pagerank is None:
                self.avg_pagerank_score = None
            else:
                vals = pagerank.values()
                self.avg_pagerank_score = sum(vals) / len(vals) if vals else None
        return self.avg_pagerank_score
    
    def get_avg_eccentricity(self) -> Optional[float]:
        if self.avg_eccentricity is None:
            eccentricity = self.get_eccentricity()
            if eccentricity is None:
                self.avg_eccentricity = None
            else:
                vals = eccentricity.values()
                self.avg_eccentricity = sum(vals) / len(vals) if vals else None
        return self.avg_eccentricity
        

    def get_degree_centrality(self) -> Dict[str, float]:
        """
        Simple measure: importance based on direct connections.
        Highly relevant for GRNs - identifies hub genes.
        Range: 0-1 (normalized by network size)
        """
        if self.degree_centrality is None:
            self.degree_centrality = nx.degree_centrality(self.graph)
        return self.degree_centrality
    
    def get_in_degree_centrality(self) -> Dict[str, float]:
        """
        Simple measure: importance based on direct connections.
        Highly relevant for GRNs - identifies hub genes.
        Range: 0-1 (normalized by network size)
        """
        if self.in_degree_centrality is None:
            self.in_degree_centrality = nx.in_degree_centrality(self.graph)
        return self.in_degree_centrality
    
    def get_out_degree_centrality(self) -> Dict[str, float]:
        """
        Simple measure: importance based on direct connections.
        Highly relevant for GRNs - identifies hub genes.
        Range: 0-1 (normalized by network size)
        """
        if self.out_degree_centrality is None:
            self.out_degree_centrality = nx.out_degree_centrality(self.graph)
        return self.out_degree_centrality

    
    def get_betweenness_centrality(self) -> Dict[str, float]:
        """
        In-between centrality: measures how often a node lies on shortest paths
        between other nodes. 
        Relevant for GRNs: identifies genes that bridge different regulatory modules.
        Range: 0-1
        """
        if self.betweenness_centrality is None:
            self.betweenness_centrality = nx.betweenness_centrality(self.graph, weight='weight')
        return self.betweenness_centrality
    
    def get_closeness_centrality(self) -> Dict[str, float]:
        """
        General centrality: average distance to all other nodes.
        Relevant for GRNs: identifies genes that can influence the network quickly.
        Range: 0-1 (higher = closer to other genes)
        """
        if self.closeness_centrality is None:
            self.closeness_centrality = nx.closeness_centrality(self.graph)
        return self.closeness_centrality
    
    def get_eigenvector_centrality(self, max_iter: int = 1000) -> Optional[Dict[str, float]]:
        """
        Identifies nodes connected to other important nodes.
        Relevant for GRNs: master regulators that control other regulators.
        Requires connected or weakly connected component for directed graphs.
        """
        if self.eigenvector_centrality is None:
            # For directed graphs, use eigenvector on undirected version
            G_undirected = self.graph.to_undirected() if self.directed else self.graph
            try:
                self.eigenvector_centrality = nx.eigenvector_centrality(G_undirected, max_iter=max_iter, weight="weight", tol=1e-4)
            except (np.linalg.LinAlgError, nx.AmbiguousSolution, nx.PowerIterationFailedConvergence) as e:
                print(f"Could not calculate eigenvectors\n{e}\n\nTrying again")
                try:
                    self.eigenvector_centrality = nx.eigenvector_centrality_numpy(G_undirected, max_iter=max_iter, weight="weight")
                except (np.linalg.LinAlgError, nx.AmbiguousSolution, nx.PowerIterationFailedConvergence) as e:
                    print(f"Could not calculate eigenvectors\n{e}\n\nCalculating for largest connected graph")
                    
                    if not nx.is_strongly_connected(G_undirected):
                        # For directed graphs
                        largest_cc = max(nx.weakly_connected_components(G_undirected), key=len)
                        G_sub = G_undirected.subgraph(largest_cc).copy()
                        self.eigenvector_centrality = nx.eigenvector_centrality(G_sub, max_iter=1000)
                    else:
                        print("Could not calculate eigenvector centrality in anyway")
                        self.eigenvector_centrality = None
        return self.eigenvector_centrality
    

    def get_katz_centrality(self, max_iter:int = 1000) -> Optional[Dict[str, float]]:
        """
        https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.katz_centrality.html#networkx.algorithms.centrality.katz_centrality
        """
        if self.katz_centrality is None:
            ## calc max eigenvalues from adjacency matrix
            adj_matrix = self.calc_adjacency_maxtix()
            alpha = 1 / max(adj_matrix) - 0.01

            try:
                self.katz_centrality = nx.katz_centrality(self.graph, alpha=alpha, max_iter=max_iter, weight="weight")
            except (np.linalg.LinAlgError, nx.AmbiguousSolution, nx.PowerIterationFailedConvergence) as e:
                print(f"could not calculate katz_centrality\n{e}")
                self.katz_centrality = None
        return self.katz_centrality

    
    def get_pagerank(self) -> Dict[str, float]:
        """
        PageRank: importance based on incoming edges from important nodes.
        Highly relevant for GRNs: identifies target genes regulated by important genes.
        Range: sum of all values = 1
        """
        if self.pagerank is None:
            self.pagerank = nx.pagerank(self.graph, weight='weight')
        return self.pagerank
    
    def get_harmonic_centrality(self) -> Dict[str, float]:
        """
        Variant of closeness: sum of reciprocals of distances.
        Better for disconnected graphs than closeness centrality.
        Relevant for GRNs: works well with disconnected regulatory modules.
        """
        if self.harmonic_centrality is None:
            self.harmonic_centrality = nx.harmonic_centrality(self.graph)
        return self.harmonic_centrality
    
    def get_eccentricity(self) -> Optional[Dict[str, float]]:
        """
        The eccentricity of a node v is the maximum distance from v to all other nodes in G.
        """
        if self.eccentricity is None:
            if not nx.is_strongly_connected(self.graph):
                self.eccentricity = None
            else:
                self.eccentricity = nx.eccentricity(self.graph)
        return self.eccentricity
    
    def get_center(self) -> Optional[List[str]]:
        """
        Returns the center of the graph G.

        The center is the set of nodes with eccentricity equal to radius.
        """
        if self.center is None:
            eccentricity = self.get_eccentricity()
            if eccentricity is None:
                self.center = None
            else:
                self.center = nx.center(self.graph, e=eccentricity)
        return self.center
    
    def get_diameter(self) -> Optional[float]:
        """
        The diameter is the maximum eccentricity.
        """
        if self.diameter is None:
            eccentricity = self.get_eccentricity()
            if eccentricity is None:
                self.diameter = None
            else:
                self.diameter = float(nx.diameter(self.graph, e=eccentricity))
        return self.diameter
    
    def get_min_weighted_dominating_set(self) -> Set[str]:
        if self.min_weighted_dominating_set is None:
            G = self.graph if not self.directed else self.graph.to_undirected()
            self.min_weighted_dominating_set = nx.algorithms.approximation.min_weighted_dominating_set(G, weight="weight")
        return self.min_weighted_dominating_set
    
    # ========== HUB TOPOLOGY MEASURES ==========
    def get_degree_assortativity(self) -> float:
        """
        Degree Assortativity Coefficient (r) - Newman 2003
        
        Quantifies tendency of nodes to connect to others with similar degrees.
        
        Returns:
            float: Assortativity coefficient between -1 and 1
                  r > 0: assortative (hubs connect to hubs)
                  r < 0: disassortative (hubs connect to low-degree nodes, clearer hub structure)
                  r ≈ 0: no preference
                  
        Note: GRNs typically show negative assortativity (hubs regulate many targets)
        """
        if self.degree_assortativity is None:
            self.degree_assortativity = float(nx.degree_assortativity_coefficient(self.graph))
        return self.degree_assortativity

    def get_degree_centralization(self) -> float:
        """
        Degree Centralization (C) - Freeman 1978
        
        Measures how strongly the network is arranged around central hub nodes.
        
        Formula: C = Σ(C_max - C(i)) / max_possible
        where C_max is the maximum degree centrality
        
        Returns:
            float: Centralization value between 0 and 1
                  0 = all nodes have equal degree (no hubs)
                  1 = star network (one central hub)
                  Higher values = clearer hub structure
        """
        if self.degree_centralization is None:
            degrees = dict(self.graph.degree())
            max_degree = max(degrees.values()) if degrees else 0
            
            # Sum of differences from maximum
            numerator = sum(max_degree - deg for deg in degrees.values())
            
            # Maximum possible sum (star network)
            n = self.graph.number_of_nodes()

            denominator = (n - 1) * (n - 1) if self.directed else (n - 1) * (n - 2)
            
            self.degree_centralization = numerator / denominator if denominator > 0 else 0.0
        return self.degree_centralization
    
    # ========== CLUSTERING ==========
    def get_average_clustering_coeff(self) -> float:
        """Average clustering coefficient across all nodes"""
        if self.average_clustering_coeff is None:
            self.average_clustering_coeff = nx.average_clustering(self.graph.to_undirected() if self.directed else self.graph)
        return self.average_clustering_coeff

    def get_large_clique_size(self) -> int:
        """
        Find the size of a large clique in a graph.

        A clique is a subset of nodes in which each pair of nodes is adjacent. This function is a heuristic for finding the size of a large clique in the graph.
        """
        if self.large_clique_size is None:
            self.large_clique_size = nx.approximation.large_clique_size(self.graph.to_undirected() if self.directed else self.graph)
        return self.large_clique_size
    
    def get_leiden_communities(self, resolution=1) -> List[Set]:
        # Note: leiden_communities doesn't cache because it depends on resolution parameter
        # If you want to cache, you'd need a dict mapping resolution -> communities
        return nx.community.leiden_communities(self.graph, resolution=resolution, weight="weight", seed=42)

    # ============ Entropy Metrics =========


    def get_shannon_vertex_entropy(self) -> Optional[float]:
        """
        Shannon Entropy of degree distribution.
        
        Measures heterogeneity/diversity of node degrees in the network.

        https://royalsocietypublishing.org/doi/pdf/10.1098/rsfs.2018.0040#:~:text=In%20particular%2C%20through%20network%20inference,community%20%5B18%2C19%5D.
        """
        if self.shannon_vertex_entropy is None:
            # Use degree distribution
            degrees = [d for _, d in self.graph.degree()]
            if not degrees:
                self.shannon_vertex_entropy = None
            else:
                # Count frequency of each degree
                from collections import Counter
                degree_counts = Counter(degrees)
                total = sum(degree_counts.values())
                distribution = [count / total for count in degree_counts.values()]
                self.shannon_vertex_entropy = self.calc_shannon_entropy(distribution=distribution)
        return self.shannon_vertex_entropy
    
    def get_structural_entropy(self) -> float:
        """
        Structural Entropy (Graph Entropy) - Li & Pan 2016
        
        Measures complexity and randomness of network structure based on
        node degree correlations and network topology.
        
        Formula: H_s = -Σ (k_i / 2m) * log2(k_i / 2m)
        where k_i is degree of node i, m is total edges
        
        Returns:
            float: Structural entropy
                  Higher values = more complex/random structure
                  Lower values = more ordered/hierarchical structure
                  
        Biological interpretation:
            - High entropy: Complex regulatory architecture
            - Low entropy: Hierarchical/ordered regulation (e.g., feed-forward loops)
            
        References:
            Li, A., & Pan, Y. (2016). Structural information and dynamical 
            complexity of networks. IEEE Trans. Information Theory.
        """
        if self.structural_entropy is None:
            n_edges = self.graph.number_of_edges()
            if n_edges == 0:
                self.structural_entropy = 0.0
            else:
                two_m = 2 *  n_edges if not self.directed else n_edges
                
                entropy = 0.0
                for node in self.graph.nodes():
                    degree = self.graph.degree(node)
                    if degree > 0:
                        p = degree / two_m
                        entropy -= p * np.log2(p)
                
                self.structural_entropy = float(entropy)
        return self.structural_entropy
    
    def get_von_neumann_entropy(self) -> Optional[float]:
        """
        Von Neumann Entropy (Quantum-inspired Graph Entropy)
        
        Based on eigenvalues of normalized graph Laplacian.
        Captures global structural properties and connectivity patterns.
        
        Formula: H_vn = -Σ λ_i * log2(λ_i)
        where λ_i are eigenvalues of normalized Laplacian
        
        Returns:
            float: Von Neumann entropy
                  Higher values = more complex connectivity patterns
                  
        Biological interpretation:
            - Sensitive to network modularity and community structure
            - High entropy: Complex, interconnected regulatory modules
            - Low entropy: Sparse, modular organization
            
        Note: Computationally expensive for large networks (O(n³))
        
        References:
            Braunstein, S. L., et al. (2006). Laplacian of a graph as a 
            density matrix. Physical Review E.
        """
        if self.von_neumann_entropy is None:
            # Filter positive eigenvalues (numerical stability)
            eigenvalues = self.eigenvalues if self.eigenvalues is not None else self.calc_eigenvalues()
            
            if len(eigenvalues) == 0:
                self.von_neumann_entropy = None
            else:
                # Normalize to form probability distribution
                eigenvalues_dist = eigenvalues / np.sum(eigenvalues)
                
                # Calculate entropy
                entropy = -np.sum(eigenvalues_dist * np.log2(eigenvalues_dist + 1e-10))
                
                self.von_neumann_entropy = float(entropy)
        return self.von_neumann_entropy
    
    def get_shannon_degree_centrality_entropy(self) -> Optional[float]:
        if self.shannon_degree_centrality_entropy is None:
            degree_centrality = self.get_degree_centrality()
            if degree_centrality is None:
                self.shannon_degree_centrality_entropy = None
            else:
                self.shannon_degree_centrality_entropy = self.get_shannon_centrality_entropy(degree_centrality.values())
        return self.shannon_degree_centrality_entropy
    
    def get_shannon_betweenness_centrality_entropy(self) -> Optional[float]:
        if self.shannon_betweenness_centrality_entropy is None:
            betweenness_centrality = self.get_betweenness_centrality()
            if betweenness_centrality is None:
                self.shannon_betweenness_centrality_entropy = None
            else:
                self.shannon_betweenness_centrality_entropy = self.get_shannon_centrality_entropy(betweenness_centrality.values())
        return self.shannon_betweenness_centrality_entropy
    
    def get_shannon_pagerank_centrality_entropy(self) -> Optional[float]:
        if self.shannon_pagerank_centrality_entropy is None:
            pagerank = self.get_pagerank()
            if pagerank is None:
                self.shannon_pagerank_centrality_entropy = None
            else:
                self.shannon_pagerank_centrality_entropy = self.get_shannon_centrality_entropy(pagerank.values())
        return self.shannon_pagerank_centrality_entropy
    
    def get_shannon_closeness_centrality_entropy(self) -> Optional[float]:
        if self.shannon_closeness_centrality_entropy is None:
            closeness_centrality = self.get_closeness_centrality()
            if closeness_centrality is None:
                self.shannon_closeness_centrality_entropy = None
            else:
                self.shannon_closeness_centrality_entropy = self.get_shannon_centrality_entropy(closeness_centrality.values())
        return self.shannon_closeness_centrality_entropy

    def get_shannon_centrality_entropy(self, centrality_values) -> float:
        """
        Entropy of centrality distribution.
        
        Measures how evenly centrality (importance) is distributed across nodes.
        
        Args:
            centrality_values: calculated from One of ['degree', 'betweenness', 'pagerank', 'closeness']
        
        Returns:
            float: Centrality entropy
                  Higher values = centrality more evenly distributed
                  Lower values = centrality concentrated on few nodes
                  
        Biological interpretation:
            - High entropy: Many genes with similar regulatory importance
            - Low entropy: Regulatory control concentrated in few master regulators
        """
        
        if not centrality_values or sum(centrality_values) == 0:
            return None
        
        # Normalize to probability distribution
        total = sum(centrality_values)
        probabilities = [c / total for c in centrality_values]
        
        # Calculate Shannon entropy
        return self.calc_shannon_entropy(distribution=probabilities)
            

    @staticmethod
    def calc_shannon_entropy(distribution: List[float]) -> float:
        """
        Classic information theory measure applied to network topology.
        
        Formula: H = -Σ p(k) * log2(p(k))
        where p(k) is the probability of degree k
        
        Args:
            distribution: Optional custom probability distribution
                         If None, uses degree distribution
        
        Returns:
            float: Shannon entropy in bits
                  Higher values = more heterogeneous degree distribution
                  Lower values = more uniform degree distribution
                  
        Biological interpretation:
            - High entropy: Diverse regulatory roles (some hubs, many low-degree nodes)
            - Low entropy: Homogeneous network (all genes similar regulatory activity)
        """
        # Calculate Shannon entropy
        entropy = 0.0
        for p in distribution:
            if p > 0:
                entropy -= p * np.log2(p)
        
        return float(entropy)
    

    # ========== INFORMATION EXCHANGE EFFICIENCY MEASURES ==========
    def get_global_efficiency(self) -> float:
        """
        Global Efficiency (Eglob) - Latora & Marchiori 2001
        
        Measures how efficiently information can be distributed across the entire network.
        Quantifies network's fault tolerance and robustness to perturbations.
        
        Formula: Eglob = (1 / (n*(n-1))) * Σ(1 / shortest_path_length(i,j))
        
        Returns:
            float: Global efficiency value between 0 and 1
                  Higher values = more efficient information exchange
        """
        if self.global_efficiency is None:
            graph_to_rv = self.graph.to_undirected() if self.directed else self.graph
            self.global_efficiency = nx.global_efficiency(graph_to_rv)
        return self.global_efficiency
    
    def get_local_efficiency(self) -> float:
        """
        Local Efficiency (Eloc) - Latora & Marchiori 2001
        
        Measures how robust the network is to small-scale perturbations.
        Quantifies how well neighbors of a node communicate when that node is removed.
        
        Formula: Eloc = (1/n) * Σ Eglob(Gi)
        where Gi is the subgraph of neighbors of node i
        
        Returns:
            float: Local efficiency value between 0 and 1
                  Higher values = more fault-tolerant to local failures

        ## takes some time to calculate (Wall time: 1min 1s on decent-size graph)
        """
        if self.local_efficiency is None:
            graph_to_rv = self.graph.to_undirected() if self.directed else self.graph
            self.local_efficiency = nx.local_efficiency(graph_to_rv)
        return self.local_efficiency
    
    #def average_shortest_path_length(self) -> float:
    #    """
    #    Average Shortest Path Length (L)
    #    
    #    Specifies average number of links needed to go from one node to another.
    #    Related to network navigability and information distribution efficiency.
    #    
    #    Note: For disconnected graphs, calculates on largest connected component.
    #    
    #    Returns:
    #       float: Average shortest path length
    #            Lower values = more efficient navigation
    #    """
    #    if self.n_nodes <= 1:
    #        return 0.0
    #    
    #    try:
    #        if self.directed:
    #            if nx.is_weakly_connected(self.graph):
    #                return nx.average_shortest_path_length(self.graph)
    #            else:
    #                # Use largest weakly connected component
    #                largest_cc = max(nx.weakly_connected_components(self.graph), key=len)
    #                subgraph = self.graph.subgraph(largest_cc)
    #                return nx.average_shortest_path_length(subgraph)
    #        else:
    #            if nx.is_connected(self.graph):
    #                return nx.average_shortest_path_length(self.graph)
    #            else:
    #                # Use largest connected component
    #                largest_cc = max(nx.connected_components(self.graph), key=len)
    #                subgraph = self.graph.subgraph(largest_cc)
    #                return nx.average_shortest_path_length(subgraph)
    #    except nx.NetworkXError:
    #        return float('inf')

    # ======  Communicibility =======

    #def communicability(self)-> Dict[Dict, str]:
    #    ### Requires Heavy computation. GPU recommended
    #    """
    #    Returns communicability between all pairs of nodes in G.

    # The communicability between pairs of nodes in G is the sum of walks of different lengths starting at node u and ending at node v.
    #
    #   Parameters
    #    :
    #    G: graph
    #    Returns
    #    :
    #    comm: dictionary of dictionaries
    #    Dictionary of dictionaries keyed by nodes with communicability as the value.

    #    Raises
    #    :
    #    NetworkXError
    #    If the graph is not undirected and simple.
    #    """
    #    
    #    return nx.communicability(self.undirected_graph)
    
    # ========== CRITICALITY MEASURES ==========
    def get_proxy_criticality_branching_ratio(self) -> float:
        """
        --> This is a proxy for the criticality branching ratio
        Normally needs time series data to be computed accurately (boolean functions)
        Criticality measure based on average branching ratio (σ).
        σ = total outgoing edges / total nodes
        Critical if σ ≈ 1: network near edge of order-chaos transition
        σ < 1: ordered (stable), σ > 1: chaotic (unstable)
        
        Relevant for GRNs: predicts network dynamics and stability
        """
        if self.proxy_criticality_branching_ratio is None:
            total_edges = self.get_n_edges()
            total_nodes = self.get_n_nodes()
            self.proxy_criticality_branching_ratio = total_edges / total_nodes if total_nodes > 0 else 0.0
        return self.proxy_criticality_branching_ratio
    
    def get_proxy_average_sensitivity(self, p: float = 0.5) -> float:
        """
        Daniels et al. 2018,
        https://arxiv.org/abs/1805.01447

        function of connectivity (k) and average activity bias of logic functions (p)

        Daniels et al. 2018 (arxiv:1805.01447)
        Average sensitivity K*p*(1-p) where:
        - K = average in-degree (connectivity)
        - p = average activity bias of Boolean functions (fraction of outputs = 1)
        
        Interpretation:
        - S < 1: ordered phase (stable)
        - S ≈ 1: critical phase (edge of chaos)
        - S > 1: chaotic phase (unstable)
        
        Highly relevant for GRNs: Boolean networks are standard models for GRNs
        --> This is a proxy for the criticality branching ratio
        Normally needs time series data to be computed accurately (boolean functions)
        """
        # Note: proxy_average_sensitivity doesn't cache because it depends on parameter p
        # If you want to cache, you'd need a dict mapping p -> sensitivity
        in_degrees = [self.graph.in_degree(node) for node in self.graph.nodes()]
        avg_in_degree = np.mean(in_degrees) if in_degrees else 0
        
        # If you have actual Boolean function data, use it; otherwise default to 0.5
        self.proxy_average_sensitivity = avg_in_degree * p * (1 - p)
        return self.proxy_average_sensitivity
    
     
