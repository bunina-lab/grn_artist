

def execute(args):
    from lib.compare_grn import CentralityMetricsComparator
    import pandas as pd
    import os
    import json

    ## Expects df to have genes on the index (1st column)
    df_con1 = pd.read_csv(args.condition_1, sep="\t", index_col=0)
    df_con2 = pd.read_csv(args.condition_2, sep="\t", index_col=0)

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    
    # Create comparator from dataframes
    comparator = CentralityMetricsComparator(
        df1=df_con1,
        df2=df_con2,
        node_column=None,  # Specify column containing node IDs
        name1 = args.name1,
        name2 = args.name2,
    )
    
    print("\n" + "="*80)
    print("ANALYZING CHANGES")
    print("="*80)
    
    results = comparator.find_all_significant_changes(
        top_n=50,
        n_permutations=1000
    )
    
    # Visualize
    comparator.visualize_all_metrics(results, top_n=10, save=os.path.join(args.outdir, "centrality_stats_comparison.png"))
    comparator.plot_centrality_heatmap(results, 
    significance_level=0.01, 
    save=os.path.join(args.outdir, "centrality_stats_heatmap.png"), 
    filter_nonsignificant_nodes=True
    )
    
    # Optionally: Write Jaccard index to a results file if desired
    with open(os.path.join(args.outdir, "jaccard_index.json"), "w") as f:
        jaccard_out = {
            "network_1": args.name1,
            "network_2": args.name2,
            "jaccard_index": comparator.jaccard_index
            }
        json.dump(jaccard_out, f, indent=4)

    if args.gdv_signature_1 and args.gdv_signature_2:
        from lib.compare_grn import GDV_compare
        gdv_comparator = GDV_compare(
            signature_file_1=args.gdv_signature_1,
            signature_file_2=args.gdv_signature_2,
            name1 = args.name1,
            name2 = args.name2,
            out_dir=args.outdir
        )
        gdv_comparator.process()
    
    print(f"Done! Check out {args.outdir}")
    


if __name__ == "__main__":
 
    import argparse
    parser = argparse.ArgumentParser(description="Compare node statistics")
    parser.add_argument("-1", "--condition_1", required=True, help="TSV file contains node statistics (rows: genes, colummns:statistics)")
    parser.add_argument("-2", "--condition_2", required=True, help="TSV file contains node statistics (rows: genes, colummns:statistics)")
    parser.add_argument("-g1", "--gdv_signature_1", required=False, default=None, help="TSV file contains graphlet degree vector (rows: genes, colummns:orbital counts)")
    parser.add_argument("-g2", "--gdv_signature_2", required=False, default=None, help="TSV file contains node statistics (rows: genes, colummns:orbital counts)")
    parser.add_argument("-n1", "--name1", required=False, default="Network_1", help="Name of the first network")
    parser.add_argument("-n2", "--name2", required=False, default="Network_2", help="Name of the second network")
    parser.add_argument("-o", "--outdir", required=True, help="output directory to generate plot files")


    args = parser.parse_args()

    execute(args)