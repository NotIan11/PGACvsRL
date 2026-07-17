import numpy as np
import matplotlib.pyplot as plt
from lqr import make_system, cost, optimal_gain, initial_gain
from reinforce import run_reinforce
from npg_sampled import run_npg

BUDGET = 50_000
SEEDS = list(range(10))
N, T, SIGMA = 10, 50, 0.5
ETAS_RF = [1e-5, 3e-5, 1e-4]                # REINFORCE grid, as tuned 
ETAS_NPG = [3e-4, 1e-3, 3e-3, 1e-2]         # NPG grid

def sweep(run_fn, etas, sys, K0):
    """
    Grid-search eta by median final gap.
    Returns best eta + (seeds x points) gaps.
    """
    best = (None, None, np.inf)

    for eta in etas:
        gaps = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            _, log = run_fn(sys, K0, eta = eta, N = N, T = T, sigma = SIGMA,
                            budget = BUDGET, rng = rng)
            gaps.append(log["gap"])

        gaps = np.array(gaps)
        med = np.median(gaps[:, -1])
        print(f"eta = {eta:g}: median = {med:.4f}, worst = {gaps[:, -1].max():.3f}")
        
        if med < best[2]:
            best = (eta, gaps, med)
    
    return best[0], best[1], log["transitions"]


def main():
    for system_name in ["reference", "reference_illcond"]:
        sys = make_system(system_name)
        K0 = initial_gain(sys)
        gap0 = cost(K0, sys) - cost(optimal_gain(sys), sys)
        print(f"\n=== {system_name} (initial gap {gap0:.2f}) ===")

        print("REINFORCE:")
        eta_rf, gaps_rf, trans = sweep(run_reinforce, ETAS_RF, sys, K0)
        print("Sampled NPG:")
        eta_npg, gaps_npg, _ = sweep(run_npg, ETAS_NPG, sys, K0)

        np.savez(f"results/exp3_{system_name}.npz",
                 transitions = trans, gaps_rf = gaps_rf, gaps_npg = gaps_npg,
                 eta_rf = eta_rf, eta_npg = eta_npg, N = N, T = T, sigma = SIGMA)
        
        plt.figure(figsize = (7, 4.5))
        for name, g, color in [(f"REINFORCE (η = {eta_rf:g})", gaps_rf, "tab:orange"),
                               (f"Sampled NPG (η = {eta_npg:g})", gaps_npg, "tab:blue")]:
            med = np.median(g, axis = 0)
            q25, q75 = np.percentile(g, [25, 75], axis = 0)
            plt.loglog(trans, med, color = color, label = name)
            plt.fill_between(trans, q25, q75, color = color, alpha = 0.25)
        
        plt.xlabel("Cumulative State Transitions")
        plt.ylabel("Cost Gap C(K) - C(K*)")
        plt.title(f"Sampled NPG vs REINFORCE - {system_name} (median, IQR, 10 seeds)")
        plt.legend(); plt.grid(True, which = "both", alpha = 0.3); plt.tight_layout()
        plt.savefig(f"plots/exp3_{system_name}.png", dpi = 150)


if __name__ == "__main__":
    main()