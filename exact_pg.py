import numpy as np
from scipy import linalg
from lqr import cost, gradient, optimal_gain, initial_gain, spectral_radius

def natural_direction(K, A, B, Q, R):
    """
    NPG Direction = vanilla gradient with the sigma_k preconditioner 
    removed from the right (Fazel et al., 2018, ICML): 2[(R + B.T @ P @ B) @ K + B.T @ P @ A]
    """
    acl = A + B @ K
    P = linalg.solve_discrete_lyapunov(acl.T, Q + K.T @ R @ K)
    return 2 * ((R + B.T @ P @ B) @ K + B.T @ P @ A)

def run_exact(sys, direction, eta, n_iters):
    """
    direction is 'vanilla' or 'natural'. Returns array of cost gaps.
    """
    A, B, Q, R = sys["A"], sys["B"], sys["Q"], sys["R"]
    K = initial_gain(sys)
    C_star = cost(optimal_gain(sys), sys)
    gaps = np.full(n_iters, np.nan)

    for i in range(n_iters):
        if direction == "vanilla":
            G = gradient(K, A, B, Q, R)
        elif direction == "natural":
            G = natural_direction(K, A, B, Q, R)
        else:
            raise ValueError(direction)
        
        K_new = K - eta * G
        if spectral_radius(A + B @ K_new) >= 1.0:
            break
        K = K_new
        gaps[i] = cost(K, sys) - C_star
    
    return gaps
