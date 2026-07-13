import numpy as np
from lqr import make_system, cost, gradient, optimal_gain, initial_gain, spectral_radius

sys = make_system()

def test_gradient(sys, K, eps=1e-6):
    G = gradient(K, sys["A"], sys["B"], sys["Q"], sys["R"])
    G_fd = np.zeros_like(K)
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            E = np.zeros_like(K); E[i,j] = eps
            G_fd[i,j] = (cost(K + E, sys) - cost(K - E, sys)) / (2 * eps)
    assert np.max(np.abs(G - G_fd)) < 1e-4

def test_exact_gd_converges(sys, eta=3e-3, n_iters=1000, tol=1e-6):
    K = initial_gain(sys)
    C_star = cost(optimal_gain(sys), sys)
    prev_gap = np.inf
    for _ in range (n_iters):
        K = K - eta * gradient(K, sys["A"], sys["B"], sys["Q"], sys["R"])
        gap = cost(K, sys) - C_star
        assert gap <= prev_gap + 1e-10
        prev_gap = gap
    assert gap < tol

test_gradient(sys, initial_gain(sys))
test_gradient(sys, 0.5 * optimal_gain(sys))
assert spectral_radius(sys["A"] + sys["B"] @ optimal_gain(sys)) < 1
test_exact_gd_converges(sys)

print("Checkpoint 1 passed:")
print(f"  C(K*)  = {cost(optimal_gain(sys), sys):.4f}   (expect ~4.4912)")
print(f"  C(K0)  = {cost(initial_gain(sys), sys):.4f}   (expect ~11.01)")
print(f"  rho(A+BK*) = {spectral_radius(sys['A'] + sys['B'] @ optimal_gain(sys)):.4f}  (expect ~0.153)")