from ast import arg
import os
import time

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
        os.environ["NETWORKX_BACKEND_PRIORITY"]="cugraph"

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    out_dir = args.outdir

    if args.simulate:
        import tempfile
        out_dir =tempfile.mkdtemp(prefix="simulation_stats_", dir=out_dir)
        assert os.path.exists(out_dir)
        
    
    timestart = time.time()
    from lib.grn_processor import GRNArtist

    grn_obj = GRNArtist(
        tsv_input = args.edge_list_tsv,
        output_dir = out_dir,
        leiden_resolution=args.leiden_resolution,
        simulate=args.simulate
    )
    grn_obj.process_grn()

    time_end = time.time()
    print(f"Took {(time_end-timestart)/60} mins")


if __name__ == "__main__":
 
    import argparse
    parser = argparse.ArgumentParser(description="GRN Artist runner")
    parser.add_argument("-i", "--edge_list_tsv", required=True, help="Path to gene - peaks or tf - motif matrix file")
    parser.add_argument("-o", "--outdir", required=True, help="output directory to generate graph and GRN statistics files")
    parser.add_argument("--leiden_resolution", required=False, default=1.0, type=float, help="Resolution for leiden clustering. Default: 1")
    parser.add_argument('--simulate', required=False, action='store_true', help="Instead of calculating all stats values of given network, calculate random network based on the given network")


    args = parser.parse_args()

    execute(args)
