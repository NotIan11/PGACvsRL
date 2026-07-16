import numpy as np
from lqr import make_system, cost, optimal_gain, initial_gain, spectral_radius
from pgac import identify, offline_stage, run_pgac

sys = make_system()
K0 = initial_gain(sys)


# Test 1: identification recovers the true model from clean-ish data
def test_identify(seed=0):
    rng = np.random.default_rng(seed)
    X0, U0, X1, _ = offline_stage(sys, K0, t0=200, sigma_e=1.0, rng=rng)
    A_hat, B_hat = identify(X0, U0, X1)
    err = np.linalg.norm(np.hstack([B_hat, A_hat]) - np.hstack([sys["B"], sys["A"]]))
    assert err < 0.5, f"identification error {err:.3f} too large"
    print(f" identify: model error = {err:.3f} (expect ~0.1-0.3")

# Test 2: offline stage produces persistently exciting data
def test_offline_pe(seed=1):
    rng = np.random.default_rng(seed)
    X0, U0, X1, _ = offline_stage(sys, K0, t0= 4 * (sys["n"] + sys["m"]), sigma_e=1.0, rng=rng)
    rank = np.linalg.matrix_rank(np.vstack([U0,X0]))
    assert rank == sys["n"] + sys["m"], f"rank {rank}, need {sys["n"] + sys["m"]}"
    print(f" offline_pe: rank - {rank} (full)")

# Test 3: short PGAC run improves the policy and stays stable
def test_pgac_short(seed=0):
    rng = np.random.default_rng(seed)
    gap0 = cost(K0, sys) - cost(optimal_gain(sys), sys)
    K, log = run_pgac(sys, K0, eta=1e-3, n_steps=1000, sigma_e=1.0, rng=rng)
    gap_final = log["gap"][-1]
    assert gap_final < 0.5 * gap0, f"gap {gap_final:.3f} not halved from {gap0:.3f}"
    assert np.all(log["rho_true"] < 1.0), "true closed loop went unstable"
    assert np.all(np.isfinite(log["gap"])), "cost hit inf (destabilized at a log point)"
    print(f" pgac_short: gap {gap0:.2f} -> {gap_final:.4f}, "
          f"max rho_true = {log['rho_true'].max():.3f}, rejected = {log['n_rejected']}")
    

if __name__ == "__main__":
    test_identify()
    test_offline_pe()
    test_pgac_short()
    print("test_pgac.py: All passed")