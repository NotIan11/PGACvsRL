import numpy as np
import scipy
from scipy import linalg
def make_system(name = "reference"):
    if name == "reference":
        Q = np.eye(4)
    elif name == "reference_illcond":
        Q = np.diag([100.0, 1.0, 1.0, 0.01])
    else:
        raise ValueError(f"Unknown system: {name}")
    
    A = np.array([[-0.13, 0.14, -0.29,  0.28],
                  [ 0.48, 0.09,  0.41,  0.30],
                  [-0.01, 0.04,  0.17,  0.43],
                  [ 0.14, 0.31, -0.29, -0.10]])
    B = np.array([[1.63, 0.93],
                  [0.26, 1.79],
                  [1.46, 1.18],
                  [0.77, 0.11]])
    n, m = B.shape
    R = np.eye(m)

    return {"A": A, "B": B, "Q": Q, "R": R, "n": n, "m": m}

def optimal_gain(sys):
    P = scipy.linalg.solve_discrete_are(sys["A"], sys["B"], sys["Q"], sys["R"])
    K_star = -np.linalg.solve(sys["R"] + sys["B"].T @ P @ sys["B"],
                              sys["B"].T @ P @ sys["A"])
    
    return K_star

def spectral_radius(M):
    return np.max(np.abs(np.linalg.eigvals(M)))

def is_stabilizing(K, sys):
    return spectral_radius(sys["A"] + sys["B"] @ K) < 1.0

def cost(K, sys):
    if not is_stabilizing(K, sys):
        return np.inf
    
    acl = sys["A"] + sys["B"] @ K
    sigma_K = scipy.linalg.solve_discrete_lyapunov(acl, np.eye(sys["n"]))

    return np.trace((sys["Q"] + K.T @ sys["R"] @ K) @ sigma_K)

def gradient(K, A, B, Q, R):
    acl = A + B @ K
    sigma_K = scipy.linalg.solve_discrete_lyapunov(acl, np.eye(A.shape[0]))
    P = scipy.linalg.solve_discrete_lyapunov(acl.T, Q + K.T @ R @ K)

    return 2 * ((R + B.T @ P @ B) @ K + B.T @ P @ A) @ sigma_K

def initial_gain(sys):
    return -optimal_gain(sys)

def rollout(K, sys, T, rng, x0 = None, sigma_u = 0.0):
    """
    Simulate the closed loop for T steps.
    
    Dynamics: x_{t+1} = A x_t + B u_t + w_t,    w_t ~ N(0,I)
    Policy:   u_t = K_t + sigma_u * eps_t,      eps_t ~ N(0,I)
              (if sigma_u = 0, deterministic policy.)
    
    Returns:
        X     : (n, T+1) states x_0, ... x_T
        U     : (m, T)   inputs u_0, ... u_{T+1}
        costs : (T,)     stage costs c_t = x_t^T Q x_t + u_t^T R u_t
    """

    A, B, Q, R = sys["A"], sys["B"], sys["Q"], sys["R"]
    n, m = sys["n"], sys["m"]

    X = np.zeros((n, T+1))
    U = np.zeros((m, T))
    costs = np.zeros(T)

    X[:, 0] = np.zeros(n) if x0 is None else x0

    for t in range(T):
        x = X[:, t]
        u = K @ x + sigma_u * rng.standard_normal(m)
        c = x @ Q @ x + u @ R @ u
        w = rng.standard_normal(n)
        X[:, t + 1] = A @ x + B @ u + w
        U[:, t] = u
        costs[t] = c
    
    return X, U, costs