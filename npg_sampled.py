import numpy as np
from lqr import cost, optimal_gain, spectral_radius
from reinforce import reinforce_gradient

def run_npg(sys, K0, eta, N, T, sigma, budget, rng, ridge=1e-6):
    """
    Sampled natural PG: REINFORCE estimate preconditioned by the inverse
    batch state covariance (Kakade's NPG specialized to the fixed-sigma
    Gaussian policy on LQR). Identical loop to run_reinforce otherwise.
    """
    K = K0.copy()
    C_star = cost(optimal_gain(sys), sys)
    n_iters = budget // (N * T)
    log = {"transitions": [], "gap": []}
    n_rejected = 0

    for it in range(n_iters):
        ghat, sigma_hat = reinforce_gradient(K, sys, N, T, sigma, rng,
                                             return_state_cov=True)
        g_nat = ghat @ np.linalg.inv(sigma_hat + ridge * np.eye(sys["n"]))
        K_new = K - eta * g_nat

        if spectral_radius(sys["A"] + sys["B"] @ K_new) < 1.0:
            K = K_new
        else:
            n_rejected += 1
            eta *= 0.5
        
        log["transitions"].append((it + 1) * N * T)
        log["gap"].append(cost(K, sys) - C_star)
    
    log["rejected"] = n_rejected
    return K, {k: np.asarray(v) if isinstance(v, list) else v for k, v in log.items()}
