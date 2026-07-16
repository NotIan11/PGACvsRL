import numpy as np
from lqr import make_system, cost, optimal_gain, initial_gain
from reinforce import run_reinforce

BUDGET = 50_000
SEEDS = range(10)
SIGMA = 0.5

# (N, T, eta) configs: eta grid at the primary batch size,
# plus one big-batch config as the sensitivity point
CONFIGS = [
    (10, 50, 1e-5),
    (10, 50, 3e-5),
    (10, 50, 1e-4),
    (50, 100, 3e-5),
]

def main():
    sys = make_system()
    K0 = initial_gain(sys)
    gap0 = cost(K0, sys) - cost(optimal_gain(sys), sys)
    print(f"initial gap = {gap0:.3f}, budget = {BUDGET} transitions\n")
    print(f"{'N':>4} {'T':>4} {'eta':>8} | {'median gap':>10} {'worst':>8} {'best':>8} {'rej':>4}")

    summary = {}
    for (N, T, eta) in CONFIGS:
        finals, n_rej = [], 0
        all_transitions, all_gaps = [], []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            _, log = run_reinforce(sys, K0, eta=eta, N=N, T=T, sigma=SIGMA, budget=BUDGET, rng=rng)
            finals.append(log["gap"][-1])
            n_rej += log["n_rejected"]
            all_transitions.append(log["transitions"])
            all_gaps.append(log['gap'])

        finals = np.array(finals)
        key = f"N{N}_T{T}_eta{eta:g}"
        summary[key] = np.median(finals)
        print(f"{N:>4} {T:>4} {eta:>8g} | {np.median(finals):>10.4f} "
              f"{finals.max():>8.3f} {finals.min():>8.4f} {n_rej:>4}")
        
        # per-config raw curves: seeds x logged points (same iter count per seed)
        np.savez(f"results/reinforce_{key}.npz",
                 transitions = np.array(all_transitions),
                 gaps = np.array(all_gaps),
                 finals = finals,
                 N = N, T = T, eta = eta, sigma = SIGMA,
                 budget = BUDGET, seeds = np.array(list(SEEDS)))
        
    best = min(summary, key=summary.get)
    print(f"\nbest config: {best} (median final gap {summary[best]:.4f})")
    print("-> use this config's npz for Experiment 2")

if __name__ == "__main__":
    main()