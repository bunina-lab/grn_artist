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




def process_database_enrichment(stats_df, graph, outdir, organism="human", top_n=10):
    ### Node enrichment
    msigdb = get_decouplr_database(organism=organism)
    enriched_stats = perform_enrichment(stats_df, msigdb, collection_filter=MSIG_DATABASE_KEYS, community_column="leiden_community")
    top_enriched_df = select_top_terms(enriched_stats, k=top_n)
    enriched_stats.to_csv(os.path.join(outdir, "enriched_terms.tsv"), index=False, sep="\t")
    top_enriched_df.to_csv(os.path.join(outdir, f"top{top_n}_enriched_terms.tsv"), index=False, sep="\t")

    ## Edge enrichment
    collectri_db = get_collectri_db(organism)
    edge_enrichment_df =process_edge_enrichment(collectri_db, graph)
    edge_enrichment_df.to_csv(os.path.join(outdir, "known_edges.tsv"), sep='\t', index=False)


    node_to_community = stats_df.loc[:,"leiden_community"].astype(int).to_dict()
    agg_graph = aggregate_graph_to_communities(graph, node_to_community)


    if any(top_enriched_df["significant"]):
        plot_enrichment_dotplot(
            enrichment_df=top_enriched_df,
            outpath=os.path.join(outdir,"terms_per_community_dotplot.png")
        )

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

    plot_graph_drawing( ## Netgraph drawing of the graph
        G=graph,
        enriched_df=top_enriched_df,
        node_stats_df=stats_df,
        outpath=os.path.join(outdir, "graph_drawing.png"),
        edge_stats_df=edge_enrichment_df,
        node2community=node_to_community,
        scaling_factor=1.2,
        only_significant=True,
        n_top_comms = 5
    )

def process_edge_enrichment(db, graph):
    edgelist_df = nx.to_pandas_edgelist(graph)
    merged = edgelist_df.merge(db, on=["source", "target"], how="left")
    merged["is_known_link"] = merged["references"].apply(lambda x :not pd.isna(x))

    edge_attrs = {}
    for _, row in merged.iterrows():
        edge = (row['source'], row['target'])
        edge_attrs[edge] = {
            'norm_weight': row['weight'],
            'sign': row['sign'],
            #'resources': row['resources'],
            'references': row['references'],
            'sign_decision': row['sign_decision'],
            'is_known_link': row['is_known_link']
        }

    nx.set_edge_attributes(graph, edge_attrs)
    return merged[merged["is_known_link"]]



def get_collectri_db(organism="human", update=False):
    ## look decoupler resource folder
    resource_path = os.path.join(DECOUPLER_RESOURCE_DIR, f"collectri_{organism}.tsv")
    if os.path.exists(resource_path) and not update:
        resoruce_pd = pd.read_csv(resource_path, sep="\t")
    else:
        resoruce_pd = dc.op.collectri(organism)
        resoruce_pd = resoruce_pd.rename({"weight":"sign"}, axis=1)
        resoruce_pd.to_csv(resource_path, sep="\t", index=False)
    return resoruce_pd

    
def get_decouplr_database(database_name="MSigDB", organism="human", update=False):
    ## look decoupler resource folder
    resource_path = os.path.join(DECOUPLER_RESOURCE_DIR, f"{database_name}_{organism}.tsv")
    if os.path.exists(resource_path) and not update:
        resoruce_pd = pd.read_csv(resource_path, sep="\t")
    else:
        resoruce_pd = dc.op.resource(database_name, organism=organism, verbose=True)
        resoruce_pd.to_csv(resource_path, sep="\t", index=False)
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

    ## LogFoldChange
    enrichment_results["logFC"] = np.log(enrichment_results["fold_enrichment"])
    
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

def plot_enrichment_dotplot(enrichment_df, outpath, top_terms=50):
    import matplotlib.pyplot as plt

    rfig = dc.pl.dotplot(
        df=enrichment_df[enrichment_df["significant"]], 
        x="community", 
        y="display_name", 
        c="-log10_p", 
        s="logFC", 
        scale=0.85, 
        top=top_terms,
        figsize=(18,12), 
        dpi=800, 
        return_fig=True
        )
    # Set the colorbar colormap to 'Reds' ## no, berlin is better
    rfig.axes[0].collections[0].colorbar.ax.collections[0].set_cmap('berlin')
    #plt.title("Endothelial"),
    plt.xticks(range(enrichment_df["community"].nunique()))
    #rfig.suptitle("Endothelial")
    plt.grid(axis = 'y')
    rfig.savefig(outpath, dpi=rfig.dpi)
    plt.close()

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
            # SOLUTION 1: Limit number of terms (top 50 by fold_enrichment)
            if len(word_freq) > 50:
                word_freq = dict(sorted(word_freq.items(), 
                                      key=lambda x: x[1], 
                                      reverse=True)[:50])
            
            # SOLUTION 2: Increase canvas size and adjust parameters
            try:
                wordcloud = WordCloud(
                    width=1600,  # Increased from 800
                    height=1200,  # Increased from 600
                    background_color='white',
                    colormap='viridis',
                    relative_scaling=0.5,
                    min_font_size=6,  # Reduced from 8
                    max_font_size=100,  # Add max font size
                    max_words=100,  # Limit total words
                    collocations=False,  # Prevent word repetition
                    prefer_horizontal=0.7  # More horizontal words fit better
                ).generate_from_frequencies(word_freq)
                
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title(f'Community {comm}\n({len(comm_terms)} significant terms)', 
                            fontsize=12, fontweight='bold')
            except ValueError as e:
                # SOLUTION 3: Fallback - if still fails, show top terms as text
                print(f"Warning: Could not generate wordcloud for community {comm}: {e}")
                top_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                text = "\n".join([f"{term}" for term, _ in top_terms])
                ax.text(0.5, 0.5, text, 
                       ha='center', va='center', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax.set_title(f'Community {comm}\n({len(comm_terms)} significant terms)\n[Top 10 shown]', 
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
        plt.savefig(out_path, dpi=600, bbox_inches='tight')
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

def plot_graph_drawing(G, enriched_df, node_stats_df, edge_stats_df, outpath, n_top_comms=3, scaling_factor=1, node2community=None, only_significant=True):
    ## 1st calculate which nodes to be plotted
    ## 2nd give enrichment terms to the plot
    ## 3rd plot with legends
    
    ## calculate best centralities:
    # Define centrality attributes for cleaner checking
    from config import CENTRALITY_COLOURS, COMMUNITY_COLOURS

    node2colour = {}
    node2size = {}
    node2labels = {}
    nodelabel_fontdict = {'size': 9}
    node2community = {} if not node2community else node2community
    node2shape = {}
    node2alpha = {}

    known_node_set = set( edge_stats_df[edge_stats_df["is_known_link"]]["source"].unique().tolist() + edge_stats_df[edge_stats_df["is_known_link"]]["target"].unique().tolist() )

    ##### Community nodes ###
    ### select most enriched communities
    top_comms= get_top_terms_w_comms_nodes(enriched_df, k=n_top_comms)
    top_comms


    selected_comms = []
    comm2enriched_terms = {}

    for com_dict in top_comms:
        comm_number = com_dict["community"]
        if comm_number not in selected_comms:
            selected_comms.append(comm_number) 
        for gene in com_dict["overlap_genes"]:
            node2colour.update({gene : COMMUNITY_COLOURS[selected_comms.index(comm_number)%len(COMMUNITY_COLOURS)] if gene not in known_node_set else 'lime'})
            node2size.update({gene : 1*scaling_factor if gene not in known_node_set else 1.5*scaling_factor})
            node2alpha[gene] = 0.7
            node2shape[gene] =  "^" if  nx.get_node_attributes(G, "is_TF")[gene] else "o"
            node2labels[gene] = gene
        
        comm2enriched_terms.setdefault(comm_number, []).append(com_dict["geneset"])
    
    comm2showingterms= {} ## {comm: terms}
    
    comm2showingterms = get_hierarchy_terms(comm2enriched_terms, level=4)
    
    community2colour_dict = {ccom :COMMUNITY_COLOURS[selected_comms.index(ccom)%len(COMMUNITY_COLOURS)] for ccom in selected_comms}
    
    ### Centrality nodes ###
    for st, clr in CENTRALITY_COLOURS.items():
        top_central_nodes = identify_top_nodes(node_stats_df[st].to_dict(), n_top=3)
        #print(st)
        #print(top_central_nodes)
        for centrl_node in top_central_nodes:
            node2colour.update({centrl_node : clr if centrl_node not in known_node_set else "lime"})
            node2size.update({centrl_node : 1.5*scaling_factor})
            node2alpha[centrl_node] = 0.9
            node2shape[centrl_node] =  "^" if nx.get_node_attributes(G, "is_TF")[centrl_node] else "o"
            node2labels[centrl_node] = centrl_node
    

    ### Check
    if len(node2colour.keys()) != len(node2community.keys()):### Take a subset of the node2
        node2community = {node:comm for node, comm in node2community.items() if node in node2colour}

    assert set(node2colour.keys()).difference(set(node2community.keys())) == set()
    
    G_sub=G.subgraph(node2colour.keys())

    ### Edge stats ###

    # 6. Prepare edge properties
    edge2colors = {}
    edge2widths = {}
    edge2alphas = {}

    for (u, v) in G_sub.edges():
        weight = G_sub[u][v]['weight']
        width = scaling_factor * weight*5
        known_link = G_sub.get_edge_data(*(u,v))["is_known_link"]
        if known_link:
            edge2colors[(u, v)] = '#27AE60'  # Green for enriched
            edge2widths[(u, v)] = width +0.25# 5
            edge2alphas[(u, v)] = 0.9
        else:
            edge2colors[(u, v)] = '#BDC3C7'  # Light gray
            edge2widths[(u, v)] = width #0.3
            edge2alphas[(u, v)] = 0.4
    

    ### Plotting script ###
    from netgraph import Graph
    import matplotlib.pyplot as plt
    # Create figure
    fig, ax = plt.subplots(figsize=(24, 18), facecolor='white', dpi=800)

    # Draw network using netgraph
    plot_instance = Graph(
        G_sub,
        node_color=node2colour,
        node_size=node2size,
        node_labels=node2labels,
        node_label_fontdict=nodelabel_fontdict,
        node_edge_width=0,
        edge_color=edge2colors,
        edge_width=edge2widths,
        arrows=True,
        ax=ax,
        node_shape=node2shape,
        node_alpha=node2alpha,
        node_layout='community', node_layout_kwargs=dict(node_to_community=node2community),
        edge_layout='bundled', edge_layout_kwargs=dict(k=2000),
    )


   # add_community_terms_to_plot(
   #     ax, 
   #     plot_instance, 
   #     community2terms=comm2showingterms, 
   #     node_to_community=node2community, 
   #     community_to_colors=community2colour_dict
   # )

   
    # Create legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

     ## Get terms patch on legend
    terms_patches = []
    for comm, terms in comm2showingterms.items():
        for term in terms:
            terms_patches.append(Patch(facecolor=community2colour_dict[comm], label=term))

    legend_elements = [
        Patch(facecolor=color, label=key.replace('top_', '').replace('_', ' ').title())
        for key, color in CENTRALITY_COLOURS.items()
    ] + [Patch(facecolor="lime", label="has known link")] + [
        Patch(facecolor=community2colour_dict[comm], label=f"Community {comm}") for comm in selected_comms
    ] + [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="black", markersize=12, linewidth=0, label="TF"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markersize=12, linewidth=0, label="Gene")
    ]

    # Add a separate legend for terms alone
    from matplotlib.legend import Legend

    terms_patches = []
    for comm, terms in comm2showingterms.items():
        for term in terms:
            terms_patches.append(Patch(facecolor=community2colour_dict[comm], label=term))

    # If there are any term patches, add a second legend for them
    if len(terms_patches) > 0:
        # Place the terms legend to the right, below the main legend
        terms_legend = plt.legend(
            handles=terms_patches,
            loc='upper left',
            bbox_to_anchor=(1, 1),
            title='Top Enriched Terms',
            fontsize=13,
            title_fontsize=14,
            frameon=False,
            ncol=1
        )
        # Add the main legend back (as adding a new legend removes previous ones)
        ax.add_artist(terms_legend)

    # Add title
    ax.set_title('Community Graph with Enriched Gene Sets', 
                fontsize=16, fontweight='bold', pad=20)

    plt.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(1, 0.05), 
            title='Top Centrality Nodes and Communities', fontsize=13, ncols=1, title_fontsize=14)
    plt.tight_layout()
    plt.savefig(outpath, dpi=800)
    



def identify_top_nodes(centrality_dict, n_top=3, method='zscore', threshold=2.0):

    """
    Identify top N nodes that are also statistically significant from the distribution.
    
    Parameters:
    -----------
    centrality_dict : dict
        Dictionary mapping node -> centrality value
    n_top : int
        Number of top nodes to consider (default 3)
    method : str
        'zscore' - use z-score threshold (default)
        'iqr' - use interquartile range (outliers)
        'percentile' - use top percentile
        'none' - just take top N without statistical test
    threshold : float
        For zscore: standard deviations above mean (default 2.0)
        For iqr: multiplier for IQR (default 1.5)
        For percentile: percentile cutoff (default 95)
    
    Returns:
    --------
    list : top N nodes that pass statistical significance test (may be < n_top)
    """
    if not centrality_dict or len(centrality_dict) == 0:
        return []
    
    values = np.array(list(centrality_dict.values()))
    nodes = list(centrality_dict.keys())
    
    # Get top N nodes by value
    sorted_indices = np.argsort(values)[::-1]
    top_n_indices = sorted_indices[:min(n_top, len(nodes))]
    top_n_nodes = [nodes[i] for i in top_n_indices]
    top_n_values = values[top_n_indices]
    
    if method == 'none':
        # Just return top N without statistical test
        return top_n_nodes
    
    # Now check which of the top N are statistically significant
    significant_nodes = []
    
    if method == 'zscore':
        # Z-score method: check if top N values are > threshold std devs above mean
        mean = np.mean(values)
        std = np.std(values)
        
        if std > 0:
            for i, node in enumerate(top_n_nodes):
                z_score = (top_n_values[i] - mean) / std
                if z_score > threshold:
                    significant_nodes.append(node)
    
    elif method == 'iqr':
        # IQR method: check if top N values are beyond Q3 + threshold * IQR
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        upper_bound = q3 + threshold * iqr
        
        for i, node in enumerate(top_n_nodes):
            if top_n_values[i] > upper_bound:
                significant_nodes.append(node)
    
    elif method == 'percentile':
        # Percentile method: check if top N values are above the percentile cutoff
        cutoff = np.percentile(values, threshold)
        
        for i, node in enumerate(top_n_nodes):
            if top_n_values[i] > cutoff:
                significant_nodes.append(node)
    
    return significant_nodes


def get_top_terms_w_comms_nodes(df, k=5):
    """ Outputs:
    [{'community': 10,
  'overlap_genes': ['ATF3', 'EGR1', 'FOS', 'FOSB', 'JUN', 'NR4A2'],
  'geneset': 'HALLMARK_TNFA_SIGNALING_VIA_NFKB'},
 {'community': 10,
  'overlap_genes': ['ATF3', 'EGR1', 'FOS', 'FOSB', 'JUN', 'NR4A2'],
  'geneset': 'GOMF_DNA_BINDING_TRANSCRIPTION_ACTIVATOR_ACTIVITY'},
 {'community': 10,
  'overlap_genes': ['FOS', 'JUN'],
  'geneset': 'REACTOME_ACTIVATION_OF_THE_AP_1_FAMILY_OF_TRANSCRIPTION_FACTORS'},
 {'community': 10,... ]
    """
    fltrd_grph = df[df["significant"]]
    community_summary = fltrd_grph.loc[fltrd_grph.groupby('community')['logFC'].idxmax()]
    top_comms = community_summary.sort_values(['logFC'], ascending=[False]).head(k)["community"].tolist()
    comm2node_dict = []

    for top_com in top_comms:
        mask = df["community"] == top_com
        for _, irow in df[mask][["community", "overlap_genes", "geneset"]].iterrows():
            # Split by comma and strip whitespace
            comm2node_dict.append({
                "community":top_com,
                "overlap_genes":[gene.strip() for gene in irow["overlap_genes"].split(",")],
                "geneset": irow["geneset"]
            })
    return comm2node_dict


def add_community_terms_to_plot(ax, plot_instance, community2terms, node_to_community, community_to_colors, only_significant=False):
    """
    Add enriched term labels to a netgraph plot based on communities
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes object with the plot
    plot_instance : netgraph.Graph
        The netgraph Graph instance
    community2terms : dict
        {comm_number:list(geneset_terms)}
    node_to_community : dict
        Mapping of nodes to their community assignments
    """
    # Get node positions from the netgraph instance
    node_positions = plot_instance.node_positions
    
    # Calculate community centroids
    communities = list(set(node_to_community.values()))
    
    community_positions = {}
    for comm in communities:
        comm_nodes = [node for node, c in node_to_community.items() if c == comm]
        if comm_nodes:
            x_coords = [node_positions[node][0] for node in comm_nodes]
            y_coords = [node_positions[node][1] for node in comm_nodes]
            centroid = (np.median(x_coords), np.median(y_coords))
            community_positions[comm] = centroid

    
    
    for comm, terms_lst in community2terms.items():
        if comm in community_positions:
            cx, cy = community_positions[comm]
            n_terms = len(terms_lst)
            
            # Place terms in a circle around the centroid
            for idx, term in enumerate(terms_lst):
                if term == "Other Pathways": continue
                angle = 2 * np.pi * idx / n_terms
                radius = 0.085  # Adjust this to control distance from centroid
                x_offset = cx + radius * np.cos(angle)
                y_offset = cy + radius * np.sin(angle)
                
                # Format the label
                #term_name = "_".join(term.split("_")[1:])
                term_name = term.title()[:35]  # Limit length
                # Add text with background
                ax.text(x_offset, y_offset, term_name, 
                       fontsize=10 if n_terms > 3 else 12,
                       fontweight='bold', #if row['significant'] else 'normal',
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.4', 
                                facecolor=community_to_colors[comm], 
                                edgecolor='black', 
                                alpha=0.6),
                       zorder=1000) 


def get_hierarchy_terms(comm2terms:dict, level=3):
    """
    will show ontologically higher terms from go obo, reactome, hallmark etc.
    """
    from lib.enriched_terms_processing import EnrichmentTermGrouper
    grouper = EnrichmentTermGrouper()
    rv = {}
    for comm, termslist in comm2terms.items():
        grouped = grouper.group_terms_from_prefixed_list(termslist, go_level=level)
        rv[comm] = list(grouped.keys())
    
    return rv
