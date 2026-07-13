import numpy as np
import matplotlib.pyplot as plt
from lqr import make_system
from exact_pg import run_exact
from utils import grid_search

ETAS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
N_ITERS = 1000

def main(system_name):
    sys = make_system(system_name)
    eta_v, gaps_v = grid_search(lambda e: run_exact(sys, "vanilla", e, N_ITERS), ETAS)
    eta_n, gaps_n = grid_search(lambda e: run_exact(sys, "natural", e, N_ITERS), ETAS)
    print(f"[{system_name}] vanilla best eta={eta_v}, natural best eta={eta_n}")

    np.savez(f"results/exp1_{system_name}.npz",
             gaps_v = gaps_v, gaps_n = gaps_n, eta_v = eta_v, eta_n = eta_n)
    
    plt.figure(figsize=(6,4))
    plt.semilogy(gaps_v, label=f"vanilla PG (η = {eta_v})")
    plt.semilogy(gaps_n, label=f"NPG (η = {eta_n})")
    plt.xlabel("iteration"); plt.ylabel("cost gap C(K) - C(K*)")
    plt.title(f"Exact PG vs NPG - {system_name}")
    plt.legend(); plt.grid(True, which="both", alpha=0.3); plt.tight_layout()
    plt.savefig(f"plots/exp1_{system_name}.png", dpi=150)

if __name__ == "__main__":
    main("reference")
    main("reference_illcond")