import numpy as np
import matplotlib.pyplot as plt
from lqr import make_system, cost, optimal_gain, initial_gain
from pgac import run_pgac

BUDGET = 50_000
SEEDS = list(range(10))
PGAC_CFG = dict(eta=1e-3, sigma_e=1.0)
REINFORCE_FILE = "results/reinforce_N10_T50_eta3e-05.npz"   # from REINFORCE sweeping, best hyperparams
THRESHOLDS = [0.10, 0.01, 0.001]                            # fractions of initial gap

def crossing(transitions, gaps, thr):
    """
    First transition count at which gap < thr, else inf.
    """
    hit = gaps < thr
    return transitions[np.argmax(hit)] if hit.any() else np.inf

def main():
    sys = make_system()
    K0 = initial_gain(sys)
    gap0 = cost(K0, sys) - cost(optimal_gain(sys), sys)

    # PGAC at full budget (each transition = one update)
    t0 = 4 * (sys["n"] + sys["m"])
    pg_t, pg_gaps = None, []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        _, log = run_pgac(sys, K0, n_steps=BUDGET - t0, rng=rng, **PGAC_CFG)
        pg_t = log["t"]
        pg_gaps.append(log["gap"])
        print(f"PGAC seed {seed}: final gap {log['gap'][-1]:.2e}, rejected {log['n_rejected']}")
    pg_gaps = np.array(pg_gaps)
    np.savez("results/pgac_budget50k.npz", transitions=pg_t, gaps=pg_gaps, seeds=np.array(SEEDS), **PGAC_CFG)


    # REINFORCE: Load sweep results (same seeds, same K0)
    rf = np.load(REINFORCE_FILE)
    rf_t, rf_gaps = rf["transitions"][0], rf["gaps"]        # (points, ), (seeds, points)

    # Figure: median + IQR band, loglog
    plt.figure(figsize=(7, 4.5))
    for name, t, g, color in [("PGAC", pg_t, pg_gaps, "tab:blue"),
                              ("REINFORCE", rf_t, rf_gaps, "tab:orange")]:
        med = np.median(g, axis=0)
        q25, q75 = np.percentile(g, [25, 75], axis=0)
        plt.loglog(t, med, color = color, label = name)
        plt.fill_between(t, q25, q75, color = color, alpha = 0.25)
    
    plt.axhline(gap0, ls = ":", c = "gray", lw = 1)
    plt.xlabel("Cumulative State Transitions")
    plt.ylabel("Cost Gap C(K) - C(K*)")
    plt.title("Sample Efficiency: PGAC vs REINFORCE (median, IQR over 10 seeds)")
    plt.legend(); plt.grid(True, which = "both", alpha = 0.3); plt.tight_layout()
    plt.savefig("plots/exp2_main.png", dpi = 150)

    # Table: median transitions to reach thresholds
    print(f"\ninitial gap = {gap0:.3f}")
    print(f"{'threshold':>12} | {'PGAC':>10} | {'REINFORCE':>10}")
    for frac in THRESHOLDS:
        thr = frac * gap0
        pg_hit = np.median([crossing(pg_t, g, thr) for g in pg_gaps])
        rf_hit = np.median([crossing(rf_t, g, thr) for g in rf_gaps])
        fmt = lambda v: "inf" if np.isinf(v) else f"{v:.0f}"
        print(f"{frac*100:>10.1f}% | {fmt(pg_hit):>10} | {fmt(rf_hit):>10}")

if __name__ == "__main__":
    main()