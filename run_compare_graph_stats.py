        
def execute(args):
    from lib.compare_grn import GraphMetricsComparator
    import os

    if not os.path.exists(args.outdir):
            os.mkdir(args.outdir)

    graphmetrics_dist_comparator = GraphMetricsComparator(
            observed_stats_json=args.graph_stats,
            simulations_dir=args.simdir,
            out_dir=args.outdir
        )
    
    graphmetrics_dist_comparator.process()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare node statistics")
    parser.add_argument("-g", "--graph_stats", required=True, help="JSON file contains graph statistics (density, assortivity etc.)")
    parser.add_argument("-o", "--outdir", required=True, help="output directory to generate plot files")
    parser.add_argument("-s", "--simdir", required=True, help="simulations directory to calculate distributions. (Alternatively other observable GRN stats may be used.)")

    args = parser.parse_args()

    execute(args)