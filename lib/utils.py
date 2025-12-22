import numpy as np
import json
import subprocess
import os

def find_files(target_dir:str, file_name:str):
    """
    recursively finds filenames of target and sub directories
    """
    path_list = call_subprocess("find", [target_dir, '-name', file_name])[1].split('\n')
    path_list = [path_str for path_str in path_list if path_str ]
    return path_list
    


def _make_serializable(obj):
    """
    Recursively process the stats dictionary to convert any non-serializable
    data (like numpy types, sets, etc.) into JSON-serializable Python types.
    """
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return sorted(_make_serializable(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return _make_serializable(obj.tolist())
    #elif hasattr(obj, 'item') and callable(obj.item):
        # e.g. numpy scalar
    #    return obj.item()
    elif obj is None:
        return None
    # catch things like inf, -inf, nan
    try:
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return str(obj)
    except Exception:
        pass
    return obj

def write_to_json(obj, file_path):
    with open(file_path, "w") as fh:
        json.dump(obj, fh, indent=4)

def read_json(file_path):
    with open(file_path, "r") as fh:
        return json.load(fh)


def call_subprocess(command: str, params: list, outfile=None, chdir=None):
    # When we want to pipe the result to a text file, then we have to use the outfile option.
    # If the program asks you to specify the output with -o etc. then leave the outfile param None
    if outfile:
        stdout_buffer = open(outfile, "wb", buffering=0)
    else:
        stdout_buffer = subprocess.PIPE

    popen_args = dict(
        args=[command] + params,
        preexec_fn=os.setsid,
        stdin=subprocess.DEVNULL,
        stdout=stdout_buffer,
        stderr=subprocess.PIPE,
        bufsize=0,
        cwd=chdir,
    )
    process = subprocess.Popen(**popen_args)
    stdout, stderr = process.communicate()

    return_code = process.returncode

    if return_code != 0:
        full_command = " ".join(popen_args['args'])
        raise Exception(full_command, stdout, stderr)
    retstdout = stdout.decode() if stdout is not None else None
    return return_code, retstdout
