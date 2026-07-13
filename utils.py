import numpy as np

def grid_search(run_fn, etas):
    """
    run_fn(eta) -> array of cost gaps (may contain trailing NaN if diverged).
    Returns (best_eta, best_gaps) by lowest final finite gap.
    """
    best_eta, best_gaps, best_final = None, None, np.inf
    for eta in etas:
        gaps = run_fn(eta)
        finite = gaps[np.isfinite(gaps)]
        final = finite[-1] if finite.size else np.inf
        if final < best_final:
            best_eta, best_gaps, best_final = eta, gaps, final
    
    return best_eta, best_gaps