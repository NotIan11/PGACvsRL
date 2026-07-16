import numpy as np
from lqr import make_system, gradient, initial_gain
from reinforce import reinforce_gradient

sys = make_system()
K0 = initial_gain(sys)

def test_estimator_direction(seed=0):
    rng = np.random.default_rng(seed)
    G_ex = gradient(K0, sys["A"], sys["B"], sys["Q"], sys["R"])
    ghat = reinforce_gradient(K0, sys, N=5000, T=200, sigma=0.5, rng=rng)
    cos = np.sum(ghat * G_ex) / (np.linalg.norm(ghat) * np.linalg.norm(G_ex))
    ratio = np.linalg.norm(ghat) / np.linalg.norm(G_ex)
    assert cos > 0.95, f"cosine {cos:.3f} too low - estimator is broken"
    print(f" estimator: cos = {cos:.3f} (expect ~0.98), |ghat|/|G_exact| = {ratio:.0f} (expect O(T), ~400)")


if __name__ == "__main__":
    test_estimator_direction()
    print("test_reinforce.py: all passed")