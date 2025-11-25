from lib.grn_processor import GRNArtist
import os

def execute(args):
    """
    Run with:
    NX_CUGRAPH_AUTOCONFIG=True
    """

    # Check CUDA availability before using nx-cugraph
    try:
        import cupy as cp
        num_devices = cp.cuda.runtime.getDeviceCount()
        if num_devices == 0:
            print("No CUDA devices found, disabling GPU backend")
            os.environ['NETWORKX_BACKEND_PRIORITY'] = ''
    except Exception as e:
        print(f"CUDA not available:\n {e}\n disabling GPU backend")
        os.environ['NETWORKX_BACKEND_PRIORITY'] = ''
    else:
        os.environ["NX_CUGRAPH_AUTOCONFIG"] = "True"
        #os.environ['NETWORKX_BACKEND_PRIORITY'] = ''

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    grn_obj = GRNArtist(
        tsv_input = args.edge_list_tsv,
        output_dir = args.outdir
    )
    grn_obj.process_grn()


if __name__ == "__main__":
 
    import argparse
    parser = argparse.ArgumentParser(description="GRN Artist runner")
    parser.add_argument("-i", "--edge_list_tsv", required=True, help="Path to gene - peaks or tf - motif matrix file")
    parser.add_argument("-o", "--outdir", required=True, help="output directory to generate graph and GRN statistics files")

    args = parser.parse_args()

    execute(args)
