import numpy as np
from lqr import cost, gradient, optimal_gain, spectral_radius

def identify(X0, U0, X1):
    """
    OLS model estimate, equation (9): [B_hat, A_hat] = X1 @ pinv([U0; X0]).
    """
    D = np.vstack([U0, X0])
    BA = X1 @ np.linalg.pinv(D)
    m = U0.shape[0]
    return BA[:, m:], BA[:, :m]

def offline_stage(sys, K0, t0, sigma_e, rng):
    """
    Collecting t0 steps of PE data with u = K0 x + e.
    """
    A, B = sys["A"], sys["B"]
    n, m = sys["n"], sys["m"]
    X0 = np.zeros((n,t0)); U0 = np.zeros((m,t0)); X1 = np.zeros((n,t0))
    x = np.zeros(n)

    for t in range(t0):
        u = K0 @ x + sigma_e * rng.standard_normal(m)
        x_next = A @ x + B @ u + rng.standard_normal(n)
        X0[:, t], U0[:, t], X1[:, t] = x, u, x_next
        x = x_next
    
    assert np.linalg.matrix_rank(np.vstack([U0,X0])) == n + m, "data not PE"
    return X0, U0, X1, x

def run_pgac(sys, K0, eta, n_steps, sigma_e, rng, t0=None, log_every=50):
    A, B, Q, R = sys["A"], sys["B"], sys["Q"], sys["R"]
    n, m = sys["n"], sys["m"]
    if t0 is None:
        t0 = 4 * (n + m)

    M = np.zeros((n + m, n + m))
    Nm = np.zeros((n, n + m))
    x = np.zeros(n)
    for t in range(t0):
        u = K0 @ x + sigma_e * rng.standard_normal(m)
        x_next = A @ x + B @ u + rng.standard_normal(n)
        d = np.concatenate([u, x])
        M += np.outer(d, d); Nm += np.outer(x_next, d)
        x = x_next
    assert np.linalg.matrix_rank(M) == n + m, "offline data not PE"

    K = K0.copy()
    C_star = cost(optimal_gain(sys), sys)
    log = {"t": [], "gap": [], "model_err": [], "rho_true": []}
    n_rejected = 0

    for t in range(n_steps):
        # interacting with the simulated true system, Alg. 1 line 3
        u = K @ x + sigma_e * rng.standard_normal(m)
        x_next = A @ x + B @ u + rng.standard_normal(n)
        d = np.concatenate([u, x])
        M += np.outer(d, d); Nm += np.outer(x_next, d)
        x = x_next
        if not np.all(np.isfinite(x)):
            raise RuntimeError(f"state diverged at t={t}; reduce eta")

        # identify, Alg. 1 line 4
        BA = np.linalg.solve(M, Nm.T).T
        B_hat, A_hat = BA[:, :m], BA[:, m:]

        # gradient step on estimated model, Alg. 1 line 5
        if spectral_radius(A_hat + B_hat @ K) < 1.0:
            K_new = K - eta * gradient(K, A_hat, B_hat, Q, R)
            if spectral_radius(A_hat + B_hat @ K_new) < 1.0:
                K = K_new
            else:
                n_rejected += 1
        else:
            n_rejected += 1

        # logging (true system quantities, for analysis only)
        if t % log_every == 0 or t == n_steps - 1:
            log["t"].append(t0 + t + 1)
            log["gap"].append(cost(K, sys) - C_star)
            log["model_err"].append(
                np.linalg.norm(np.hstack([B_hat, A_hat]) - np.hstack([B,A]))
                )
            log["rho_true"].append(spectral_radius(A + B @ K))
    
    log["n_rejected"] = n_rejected
    return K, {k: np.asarray(v) if isinstance(v, list) else v for k, v in log.items()}

        