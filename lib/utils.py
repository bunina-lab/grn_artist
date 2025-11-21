import numpy as np

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