from grn_processor import GRNArtist

def execute(args):
    grn_obj = GRNArtist(
        tsv_input = args.edge_list_tsv,
        output_dir = args.outdir
    )
    grn_obj.initialise_graph()
    grn_obj.process_grn_statistics()




if __name__ == "__main__":
 
    import argparse
    parser = argparse.ArgumentParser(description="GRN Artist runner")
    parser.add_argument("-i", "--edge_list_tsv", required=True, help="Path to gene - peaks or tf - motif matrix file")
    parser.add_argument("-o", "--outdir", required=True, help="output directory to generate graph and GRN statistics files")
    parser.add_argument(
        "-t",
        "--filter-threshold",
        type=float,
        default=0.2,
        help="Filter threshold value (float)",
    )
    parser.add_argument(
        "-s",
        "--significance-threshold",
        type=float,
        default=0.05,
        help="Significance threshold value (float)",
    )
    args = parser.parse_args()

    execute(args)
