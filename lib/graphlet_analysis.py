import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import os


class GraphletAnalyzer:
    def __init__(self, graphlet_matrix, id_to_name=None, out_dir=None):
        """
        Initialize analyzer with graphlet count matrix
        
        Args:
            graphlet_matrix: numpy array of shape (n_nodes, n_graphlet_types)
            id_to_name: optional dictionary mapping node IDs to names
        """
        self.matrix = np.array(graphlet_matrix)
        self.id_to_name = id_to_name #if id_to_name is not None else {}
        assert self.matrix.shape[0] == len(self.id_to_name); "Matrix shape and nodes len do not match"
        assert self.matrix.shape[1] == 73; "Matrix shape and graphlet numbers do not match"

        self.n_nodes = len(self.id_to_name)
        self.node_ids = list(self.id_to_name.keys()) #node_ids if node_ids is not None else list(range(self.n_nodes))

        self.graphlet_names = None 
        self.labels = None
         
        self.out_dir = out_dir
        
        # Normalize to get Graphlet Degree Distribution (GDD) signatures
        self.signatures = None
        self.column_mask = None  # Track which columns were kept after filtering

        ## Similarity matrix
        self.similarity_matrix = None # (n_node X n_node)

    def initialise_analyser(self):
         self.graphlet_names = [f"G{i}" for i in range(self.matrix.shape[1])]
         self.labels = [self.get_node_name(self.node_ids[i]) for i in range(self.n_nodes)]



    def get_signatures(self):
        if self.signatures is None:
            self.preprocess_gdv_matrix()
        return self.signatures


    def preprocess_gdv_matrix(self):
        ## Filter zero columns
        #mask = self.matrix.sum(axis=0) > 0
        self.column_mask = np.ones(self.matrix.shape[1], dtype=bool) #mask  # Store which columns were kept
        #fltrd_gdv_matrix = self.matrix[:, mask]

        ## Normalise graphlet counts
        def normalize_rows(X):
            ## outputs unit vectors
            n = np.linalg.norm(X, axis=1, keepdims=True) ##euclidean length
            return X / np.clip(n, 1e-12, None)

        normalised_vector = normalize_rows(self.matrix)#np.log1p(normalize_rows(self.matrix))
        self.signatures = normalised_vector #/ normalised_vector.sum(axis=0)
        
        # Update graphlet_names to match filtered columns
        #if self.graphlet_names is not None:
        #    self.graphlet_names = [self.graphlet_names[i] for i in range(len(self.graphlet_names)) if mask[i]]
        
        return self.signatures

    def _finalize_plot(self, filename):
        """Save plot to disk when out_dir is set, otherwise display it."""
        if self.out_dir and filename:
            os.makedirs(self.out_dir, exist_ok=True)
            plt.savefig(os.path.join(self.out_dir, filename), bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def get_node_name(self, node_id):
        """Get node name from ID, returns ID if name not found"""
        return self.id_to_name.get(node_id, node_id)
    
    def compute_similarity(self, metric='cosine'):
        """
        Compute pairwise similarity between node signatures
        
        Args:
            metric: 'cosine' or 'euclidean'
        
        Returns:
            similarity matrix (higher = more similar for cosine)
        """
        if metric == 'cosine':
            self.similarity_matrix = cosine_similarity(self.signatures)
        elif metric == 'euclidean':
            # Convert distance to similarity
            dist = euclidean_distances(self.signatures)
            self.similarity_matrix = 1 / (1 + dist)
        else:
            raise ValueError("Metric must be 'cosine' or 'euclidean'")
        return self.similarity_matrix
    
    def compare_similarity(self, node_ids:list):
        return self.similarity_matrix[np.ix_(node_ids, node_ids)]


    def find_most_similar(self, node_idx, k=5, metric='cosine'):
        """
        Find k most similar nodes to a given node
        
        Args:
            node_idx: index of the query node
            k: number of similar nodes to return
            metric: similarity metric to use
        
        Returns:
            list of (node_id, node_name, similarity_score) tuples
        """
        similarity = self.compute_similarity(metric)
        node_similarities = similarity[node_idx]
        
        # Get indices of k+1 most similar (excluding the node itself)
        similar_indices = np.argsort(node_similarities)[::-1][1:k+1]
        
        results = [(self.node_ids[idx], 
                   self.get_node_name(self.node_ids[idx]),
                   node_similarities[idx]) 
                   for idx in similar_indices]
        return results
    
    def plot_similarity_heatmap(self, max_nodes=50, use_names=True):
        """Plot heatmap of node similarities"""
        similarity = self.compute_similarity()
        
        # Limit to max_nodes for readability
        n = min(max_nodes, self.n_nodes)
        
        # Create labels (names or IDs)
        if use_names:
            labels = [self.get_node_name(self.node_ids[i]) for i in range(n)]
        else:
            labels = self.node_ids[:n]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(similarity[:n, :n], cmap='viridis', 
                    xticklabels=labels,
                    yticklabels=labels)
        plt.title('Node Similarity Heatmap (Cosine Similarity)')
        plt.xlabel('Node')
        plt.ylabel('Node')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        self._finalize_plot("similarity_heatmap.png")
    
    
    def plot_signature_comparison(self, node_indices, use_names=True, save_plot_name=None):
        """
        Plot and compare graphlet signatures of specific nodes
        
        Args:
            node_indices: list of node indices to compare
            use_names: if True, use node names in legend
        """
        
        x = np.arange(len(self.graphlet_names))
        width = 0.8 / len(node_indices)
        
        plt.figure(figsize=(12, 6))
        for i, node_idx in enumerate(node_indices):
            offset = width * (i - len(node_indices)/2 + 0.5)
            node_id = self.node_ids[node_idx]
            label = self.get_node_name(node_id) if use_names else node_id
            plt.bar(x + offset, self.signatures[node_idx], width,
                   label=f'{label}')
        
        plt.xlabel('Graphlet Type')
        plt.ylabel('Normalized Frequency')
        plt.title('Graphlet Signature Comparison')
        plt.xticks(x, self.graphlet_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        self._finalize_plot(save_plot_name)
    
    def plot_pca_projection(self, n_clusters=None, annotate_nodes=None, use_names=True):
        """
        Project signatures to 2D using PCA and plot
        
        Args:
            n_clusters: if provided, perform k-means clustering
            annotate_nodes: list of node indices to annotate on plot
            use_names: if True, use node names for annotations
        """
        pca = PCA(n_components=2)
        projected = pca.fit_transform(self.signatures)
        
        plt.figure(figsize=(12, 9))
        
        if n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(self.signatures)
            scatter = plt.scatter(projected[:, 0], projected[:, 1], 
                                c=clusters, cmap='tab20', alpha=0.6, s=50)
            plt.colorbar(scatter, label='Cluster')
        else:
            plt.scatter(projected[:, 0], projected[:, 1], alpha=0.6, s=50)
        
        # Annotate specific nodes if requested
        if annotate_nodes:
            for idx in annotate_nodes:
                node_id = self.node_ids[idx]
                label = self.get_node_name(node_id) if use_names else node_id
                plt.annotate(label, (projected[idx, 0], projected[idx, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.8)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('Node Signatures in 2D (PCA Projection)')
        plt.tight_layout()
        self._finalize_plot("pca_projection.png")
        
        return projected, clusters if n_clusters else None
    
    def plot_similar_nodes_network(self, node_idx, k=5, use_names=True):
        """
        Visualize a node and its k most similar nodes
        """
        similar = self.find_most_similar(node_idx, k)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Node similarities
        query_node_id = self.node_ids[node_idx]
        query_label = self.get_node_name(query_node_id) if use_names else query_node_id
        
        nodes = [query_label] + [name if use_names else node_id 
                                 for node_id, name, _ in similar]
        similarities = [1.0] + [s for _, _, s in similar]
        
        colors = plt.cm.RdYlGn(similarities)
        ax1.barh(range(len(nodes)), similarities, color=colors)
        ax1.set_yticks(range(len(nodes)))
        ax1.set_yticklabels(nodes)
        ax1.set_xlabel('Similarity Score')
        ax1.set_title(f'Top {k} Most Similar Nodes to {query_label}')
        ax1.set_xlim([0, 1])
        
        # Right: Signature comparison
        indices = [node_idx] + [self.node_ids.index(node_id) for node_id, _, _ in similar]
        graphlet_names = [f'G{i}' for i in range(self.signatures.shape[1])]
        
        x = np.arange(len(graphlet_names))
        width = 0.15
        
        for i, idx in enumerate(indices):
            offset = width * (i - len(indices)/2 + 0.5)
            node_id = self.node_ids[idx]
            label = self.get_node_name(node_id) if use_names else node_id
            ax2.bar(x + offset, self.signatures[idx], width,
                   label=f'{label}', alpha=0.8)
        
        ax2.set_xlabel('Graphlet Type')
        ax2.set_ylabel('Normalized Frequency')
        ax2.set_title('Signature Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(graphlet_names)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        self._finalize_plot("similar_nodes.png")

    def plot_cluster_overlay(self, alpha=0.15, save_plot_name="cluster_overlay.png"):
        plt.figure(figsize=(12, 6))

        n_tfs, n_graphlets = self.signatures.shape
        unique_labels = sorted(set(self.labels))

        # assign each cluster a color
        # matplotlib colormaps expect numeric indices, so map labels -> indices
        cmap = plt.cm.get_cmap("tab20", len(unique_labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        colors = {label: cmap(label_to_idx[label]) for label in unique_labels}

        for i in range(n_tfs):
            cluster = self.labels[i]
            color = colors[cluster]
            plt.plot(
                np.arange(n_graphlets),
                self.signatures[i],
                color=color,
                alpha=alpha,
                linewidth=1
            )
        
        plt.xticks(np.arange(n_graphlets), self.graphlet_names, rotation=90)

        ##plt.title("AgglomerativeClustering(metric=cosine)")
        plt.tight_layout()
        self._finalize_plot(save_plot_name)
