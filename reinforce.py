import numpy as np
from lqr import cost, optimal_gain, spectral_radius

def reinforce_gradient(K, sys, N, T, sigma, rng, return_state_cov=False):
    """
    Score-function gradient estimate with reward-to-go and baseline.
    Policy: u = Kx + sigma * eps.
    Returns ghat, shape (m,n).
    """
    A, B, Q, R = sys["A"], sys["B"], sys["Q"], sys["R"]
    n, m = sys["n"], sys["m"]

    X = np.zeros((N, T + 1, n)); U = np.zeros((N, T, m)); c = np.zeros((N, T))
    X[:, 0, :] = rng.standard_normal((N, n))
    for t in range(T):
        x = X[:, t, :]
        u = x @ K.T + sigma * rng.standard_normal((N,m))
        c[:, t] = np.einsum('ij,jk,ik->i', x, Q, x) + np.einsum('ij,jk,ik->i', u, R, u)
        X[:, t + 1, :] = x @ A.T + u @ B.T + rng.standard_normal((N,n))
        U[:, t, :] = u
    
    G = np.cumsum(c[:, ::-1], axis=1)[:, ::-1]      # reward-to-go G_t, (N, T)
    adv = G - G.mean(axis=0, keepdims= True)        # baseline: per-t mean over trajectories

    resid = (U - X[:, :T, :] @ K.T) / sigma**2      # grad log pi factor, (N, T, m)
    ghat = np.einsum('itm,itn,it->mn', resid, X[:, :T, :], adv) / N

    if return_state_cov:
        sigma_hat = np.einsum('itm,itn->mn', X[:, :T, :], X[:, :T, :]) / (N * T)
        return ghat, sigma_hat
    return ghat

def run_reinforce(sys, K0, eta, N, T, sigma, budget, rng):
    """
    Optimize until the transition budget is spent.
    One iteration costs N*T transitions.
    """
    K = K0.copy()
    C_star = cost(optimal_gain(sys), sys)
    n_iters = budget // (N * T)
    log = {"transitions": [], "gap": []}
    n_rejected = 0

    for it in range(n_iters):
        ghat = reinforce_gradient(K, sys, N, T, sigma, rng)
        K_new = K - eta * ghat
        if spectral_radius(sys["A"] + sys["B"] @ K_new) < 1.0:
            K = K_new
        else:
            n_rejected += 1
            eta *= 0.5
        log["transitions"].append((it + 1) * N * T)
        log["gap"].append(cost(K, sys) - C_star)
    
    log["n_rejected"] = n_rejected
    return K, {k: np.asarray(v) if isinstance(v, list) else v for k, v in log.items()}