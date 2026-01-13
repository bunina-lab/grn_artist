## This script will overlap the nodes (which would be TFs and genes) to decoupler databases with the decoupler tool

"""
decouplR paper

https://academic.oup.com/bioinformaticsadvances/article/2/1/vbac016/6544613?login=true
"""

import decoupler as dc
import pandas as pd
import numpy as np
from config import MSIG_DATABASE_KEYS, DECOUPLER_RESOURCE_DIR
from scipy.stats import hypergeom
import matplotlib.pyplot as plt
import os
import networkx as nx


# using msigdb
#msigdb = dc.op.get_resource("MSigDB")

## from version 2.0.0
### https://decoupler.readthedocs.io/en/latest/notebooks/scell/rna_sc.html


def calc_db_score(db, adata, tmin=5, verbose=False):
    """
    Calculates scores with different methods:
    ora, ulm, mlm
    Adds score_* to .obsm inplace
    """
    try:
        dc.mt.mlm(adata, net=db, verbose=verbose, tmin=tmin)
    except Exception as e:
        print("multivariate fitting does not work:\n{error}".format(error=e))
    dc.mt.ulm(adata, net=db, verbose=verbose, tmin=tmin)
    dc.mt.ora(adata, net=db, verbose=verbose, tmin=tmin)

## input: msigdb pandas dataframe
#	genesymbol	collection	geneset
#0	A1BG	reactome_pathways	REACTOME_HEMOSTASIS
#1	A1BG	go_cellular_component	GOCC_PLATELET_ALPHA_GRANULE_LUMEN
#2	A1BG	chemical_and_genetic_perturbations	CHENG_IMPRINTED_BY_ESTRADIOL
#3	A1BG	immunesigdb	GSE13522_WT_VS_IFNG_KO_SKING_T_CRUZI_Y_STRAIN_...
#4	A1BG	immunesigdb	GSE25088_CTRL_VS_IL4_AND_ROSIGLITAZONE_STIM_MA...
#...	...	...	...
#5895457	ZZZ3	mirna_targets_mirdb	MIR656_3P
#5895458	ZZZ3	mirna_targets_mirdb	MIR513B_5P
#5895459	ZZZ3	mirna_targets_mirdb	MIR449C_5P
#5895460	ZZZ3	tf_targets_gtrf	SETD7_TARGET_GENES
#5895461	ZZZ3	mirna_targets_mirdb	MIR4719
#5895462 rows × 3 columns


## input: genes with graph statistics
#       degree_centrality	is_in_dominating_set    leiden_cluster
#ESRRG	0.133315	True    1
#A2M	0.000557	False   2
#ZNF331	0.107988	True    1
#PPARG	0.193710	True    3
#AADACL3	0.000278	False   1
#...	...	... 
#ZNF563	0.000278	False   3
#ZNF641	0.001113	False   2
#ZNF765	0.000557	False   2
#ZNF792	0.004453	False   2
#ZNF821	0.002783	False   1



def process_database_enrichment(stats_df, graph, outdir, organism="human"):
    msigdb = get_decouplr_database(organism=organism)
    enriched_stats = perform_enrichment(stats_df, msigdb, collection_filter=MSIG_DATABASE_KEYS, community_column="leiden_community")
    top_enriched_df = select_top_terms(enriched_stats, k=5)
    top_enriched_df.to_csv(os.path.join(outdir, "top_5_enriched_terms.tsv"), index=False, sep="\t")

    node_to_community = stats_df.loc[:,"leiden_community"].astype(int).to_dict()
    agg_graph = aggregate_graph_to_communities(graph, node_to_community)

    plot_community_graph_with_terms(
        G=agg_graph, 
        pos=nx.circular_layout(agg_graph), 
        top_terms=top_enriched_df, 
        node_to_community={_id:_id for _id in range(agg_graph.number_of_nodes())},
        out_path=os.path.join(outdir, "community_agg_graph_top_term.png")
        )

    create_community_wordclouds(
        enriched_stats,
        out_path=os.path.join(outdir, "enriched_terms_per_community.png")
    )



    
def get_decouplr_database(database_name="MSigDB", organism="human", update=False):
    ## look decoupler resource folder
    if os.path.exists(os.path.join(DECOUPLER_RESOURCE_DIR, f"{database_name}_{organism}.tsv")) and not update:
        resoruce_pd = pd.read_csv(os.path.join(DECOUPLER_RESOURCE_DIR, f"{database_name}_{organism}.tsv"), sep="\t")
    else:
        resoruce_pd = dc.op.resource(database_name, organism=organism, verbose=True)
        resoruce_pd.to_csv(os.path.join(DECOUPLER_RESOURCE_DIR, f"{database_name}_{organism}.tsv"), sep="\t", index=False)
    return resoruce_pd


def perform_enrichment(stats_df, enrichment_db, collection_filter=None, community_column='leiden_community', p_correction='bh'):
    """
    Perform hypergeometric test for enrichment of gene sets in each community.
    
    Parameters:
    -----------
    stats_df : DataFrame with genes as index and 'community_column' column
    enrichment_db : DataFrame with columns ['genesymbol', 'collection', 'geneset']
    collection_filter : str or list, filter to specific collection(s) (e.g., 'reactome_pathways')
    top_n : int, number of top enriched terms to keep per community
    
    Returns:
    --------
    enrichment_results : DataFrame with enrichment statistics
    """
    
    # Filter enrichment database by collection if specified
    if collection_filter is not None:
        if isinstance(collection_filter, str):
            collection_filter = [collection_filter]
        enrichment_db = enrichment_db[enrichment_db['collection'].isin(collection_filter)]
    
    # Get all genes in the network (background)
    all_genes = set(stats_df.index)
    N = len(all_genes)  # Total genes in background
    
    results = []
    
    # For each community
    for community_id in stats_df[community_column].dropna().unique():
        # Get genes in this community
        community_genes = set(stats_df[stats_df[community_column] == community_id].index)
        n = len(community_genes)  # Genes in community
        
        # For each gene set in the enrichment database
        for (collection, geneset), group in enrichment_db.groupby(['collection', 'geneset']):
            geneset_genes = set(group['genesymbol'])
            
            # Only consider gene sets that have genes in our background
            geneset_genes_in_background = geneset_genes.intersection(all_genes)
            K = len(geneset_genes_in_background)  # Genes in gene set
            
            if K == 0:  # Skip if no overlap with background
                continue
            
            # Calculate overlap
            overlap = community_genes.intersection(geneset_genes_in_background)
            k = len(overlap)  # Observed overlap
            
            if k == 0:  # Skip if no overlap
                continue
            
            # Hypergeometric test: p-value for enrichment
            # P(X >= k) where X ~ Hypergeom(N, K, n)
            p_value = hypergeom.sf(k - 1, N, K, n)
            
            # Calculate enrichment metrics
            expected = (n * K) / N
            fold_enrichment = k / expected if expected > 0 else 0
            
            results.append({
                'community': int(community_id),
                'collection': collection,
                'geneset': geneset,
                'p_value': p_value,
                'overlap_size': k,
                'community_size': n,
                'geneset_size': K,
                'fold_enrichment': fold_enrichment,
                'overlap_genes': ','.join(sorted(overlap))
            })
    
    # Create results dataframe
    enrichment_results = pd.DataFrame(results)
    
    if len(enrichment_results) == 0:
        print("No enrichment found!")
        return enrichment_results
    
    # Calculate adjusted p-values (Bonferroni)
    if p_correction == "bonferroni":
        enrichment_results['p_adjusted'] = enrichment_results['p_value'] * len(enrichment_results)
        enrichment_results['p_adjusted'] = enrichment_results['p_adjusted'].clip(upper=1.0)
    
    ## Benjamini-Hochberg
    elif p_correction.lower() == "bh" or p_correction.lower() == "benjamini-hochberg":
        from scipy import stats
        enrichment_results['p_adjusted'] = stats.false_discovery_control(enrichment_results['p_value'], method='bh')
    
    else:
        raise ValueError(f"Unknown p_correction method:\n{p_correction}\nExpected: bonferroni, bh")
    
    # Add -log10(p_value) for plotting
    enrichment_results['-log10_p'] = -np.log10(enrichment_results['p_value'])
    
    # Sort and get top N per community
    enrichment_results = enrichment_results.sort_values('p_value')
    #top_results = enrichment_results.groupby('community').head(top_n)
    
    return enrichment_results


def select_top_terms(df, k=1):
    """
    Select the most enriched term for each community where p_adjusted < 1.0
    """
    # Filter for p_adjusted < 1.0
    filtered_df = df[df['p_value'] < 5e-2].copy()
    
    # Sort by community and -log10_p (descending)
    filtered_df = filtered_df.sort_values(['community', '-log10_p'], ascending=[True, False])
    
    # Get the top term for each community
    top_terms = filtered_df.groupby('community').head(k).reset_index()
    
    # Add significance marker
    top_terms['significant'] = top_terms['p_adjusted'] < 0.05
    def _format_display_name(row):
        # Ensure we always return a scalar string to avoid pandas expanding into multiple columns
        geneset = str(row['geneset'])
        parts = geneset.split('_')
        trimmed = '_'.join(parts[1:]) if len(parts) > 1 else geneset
        suffix = ' *' if row['significant'] else ''
        return f"{trimmed}{suffix}"

    top_terms['display_name'] = top_terms.apply(_format_display_name, axis=1)
    
    return top_terms


def create_community_wordclouds(df, n_cols=3, out_path=None):
    """
    Create word clouds for each community showing significantly enriched terms
    
    Parameters:
    -----------
    df : pd.DataFrame
        Your enrichment dataframe
    n_cols : int
        Number of columns in the subplot grid
    """
    from wordcloud import WordCloud

    # Filter for significant terms only
    sig_df = df[df['p_adjusted'] < 0.05].copy()
    
    # Get unique communities
    communities = sorted(sig_df['community'].unique())
    n_communities = len(communities)
    
    if n_communities == 0:
        print("No significant terms found (p_adjusted < 0.05)")
        return None
    
    # Calculate grid dimensions
    n_rows = int(np.ceil(n_communities / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_communities == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, comm in enumerate(communities):
        ax = axes[idx]
        
        # Get terms for this community
        comm_terms = sig_df[sig_df['community'] == comm].copy()
        
        # Create word frequency dictionary (use fold_enrichment as weight)
        word_freq = {}
        for _, row in comm_terms.iterrows():
            term = "_".join(row['geneset'].split("_")[1:])
            term = term.title()
            # Use fold_enrichment as frequency weight
            word_freq[term] = row['fold_enrichment']
        
        if word_freq:
            # Create word cloud
            wordcloud = WordCloud(width=800, height=600, 
                                 background_color='white',
                                 colormap='viridis',
                                 relative_scaling=0.5,
                                 min_font_size=8).generate_from_frequencies(word_freq)
            
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(f'Community {comm}\n({len(comm_terms)} significant terms)', 
                        fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No significant terms', 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'Community {comm}', fontsize=12, fontweight='bold')
        
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_communities, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Significantly Enriched Gene Sets by Community (p_adjusted < 0.05)', 
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path)
        plt.close()
    else:
        plt.show()


def plot_community_graph_with_terms(G, pos, top_terms, node_to_community, edge_weight_attr='weight', out_path=None):
    """
    Plot a community graph with enriched terms labeled
    
    Parameters:
    -----------
    G : networkx.Graph
        Your graph object
    pos : dict
        Position dictionary for nodes
    top_terms : pd.DataFrame
        DataFrame with top terms per community
    node_to_community : dict
        Mapping of nodes to their community assignments
    edge_weight_attr : str
        Name of the edge attribute containing weights (default='weight')
    """
    fig, ax = plt.subplots(figsize=(32, 24))
    
    # Create community color map
    communities = list(set(node_to_community.values()))
    colors = plt.cm.tab20(np.linspace(0, 1, len(communities)))
    community_colors = {comm: colors[i] for i, comm in enumerate(communities)}
    
    # Color nodes by community
    node_colors = [community_colors[node_to_community[node]] for node in G.nodes()]
    
    # Draw edges with varying width/alpha based on weight
    edges = G.edges()
    if len(edges) > 0:
        # Get edge weights
        if edge_weight_attr in G.edges[list(edges)[0]]:
            weights = [G.edges[u, v].get(edge_weight_attr, 1) for u, v in edges]
            # Normalize weights for visualization
            max_weight = max(weights) if max(weights) > 0 else 1
            min_weight = min(weights)
            
            # Create width and alpha values based on weights
            widths = [0.5 + 4.0 * (w - min_weight) / (max_weight - min_weight + 1e-10) 
                     for w in weights]
            alphas = [0.1 + 0.6 * (w - min_weight) / (max_weight - min_weight + 1e-10) 
                     for w in weights]
            
            # Draw edges individually with different alphas/widths
            for (u, v), width, alpha in zip(edges, widths, alphas):
                nx.draw_networkx_edges(G, pos, [(u, v)], 
                                      width=width, alpha=alpha, ax=ax)
        else:
            # If no weights, draw all edges uniformly
            nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=100, alpha=0.7, ax=ax)
    
    # Draw node labels (original names)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', 
                           font_color='black', ax=ax)
    
    # Calculate community centroids and bounding boxes for label placement
    community_positions = {}
    community_bounds = {}
    for comm in communities:
        comm_nodes = [node for node, c in node_to_community.items() if c == comm]
        if comm_nodes:
            x_coords = [pos[node][0] for node in comm_nodes]
            y_coords = [pos[node][1] for node in comm_nodes]
            centroid = (np.mean(x_coords), np.mean(y_coords))
            community_positions[comm] = centroid
            # Calculate the spread of the community for smart positioning
            community_bounds[comm] = {
                'x_range': max(x_coords) - min(x_coords),
                'y_range': max(y_coords) - min(y_coords),
                'centroid': centroid
            }
    
    # Add term labels with smart positioning to avoid overlap
    label_spacing = 0.05  # Vertical spacing between labels (adjust as needed)
    
    # Group terms by community
    community_terms = top_terms.groupby('community')
    
    for comm, group in community_terms:
        if comm in community_positions:
            cx, cy = community_positions[comm]
            n_terms = len(group)
            
            # Calculate starting position (centered vertically)
            #y_start = cy + (n_terms - 1) * label_spacing / 2
            
            # Plot each term with offset
            for idx, (_, row) in enumerate(group.iterrows()):
                angle = 2 * np.pi * idx / n_terms
                x_offset = cx + 0.25 * np.cos(angle)
                y_offset = cy + 0.25 * np.sin(angle)
                # Calculate vertical offset
                #y_offset = y_start - idx * label_spacing
                
                # Optional: add slight horizontal jitter for very close labels
                #x_offset = cx
                #if n_terms > 3:
                    # Alternate left/right for many labels
                 #   x_offset += (0.03 if idx % 2 == 0 else -0.03)
                
                # Format the label
                term_name = "_".join(row['geneset'].split("_")[1:])
                term_name = term_name.title()[:45]  # Limit length
                if row['significant']:
                    term_name += ' *'
                
                # Add text with background
                ax.text(x_offset, y_offset, term_name, 
                       fontsize=9 if n_terms > 2 else 10,  # Smaller font for many labels
                       fontweight='bold' if row['significant'] else 'normal',
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.4', 
                                facecolor=community_colors[comm], 
                                edgecolor='black', 
                                alpha=0.8))
    
    ax.set_title('Community Graph with Enriched Gene Sets\n(* indicates p_adjusted < 0.05)', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path)
        plt.close()
    else:
        plt.show()

### Aggregate plot 

def aggregate_graph_to_communities(graph, node_to_community):
    # Create a new graph where each community is represented as a single node
    aggregated_graph = nx.Graph()
    community_map = {}

    # Create a mapping from community to a unique node in the aggregated graph
    for node, community in node_to_community.items():
        if community not in community_map:
            community_map[community] = len(community_map)
            aggregated_graph.add_node(community_map[community])

    # Add edges between communities if they are connected in the original graph
    # We'll sum weights for edges between different communities.
    edge_weights = {}

    for edge in graph.edges(data=True):
        node1, node2 = edge[0], edge[1]
        data = edge[2] if len(edge) > 2 else {}

        community1 = community_map[node_to_community[node1]]
        community2 = community_map[node_to_community[node2]]

        # Only add edges between different communities
        if community1 != community2:
            # Always order tuple for undirected graphs
            comm_pair = tuple(sorted((community1, community2)))
            
            weight = data.get('weight', 1.0)  # Default to weight 1 if not present
            edge_weights[comm_pair] = edge_weights.get(comm_pair, 0.0) + weight

    # Now, add aggregated edges with summed weights to the community graph
    for (comm1, comm2), total_weight in edge_weights.items():
        aggregated_graph.add_edge(comm1, comm2, weight=total_weight)
    
    return aggregated_graph