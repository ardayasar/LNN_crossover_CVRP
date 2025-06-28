#!/usr/bin/env python3
"""
speed_quality_pareto.py

Pareto‐style speed–quality scatter for LNN on E-n22-k4 (with mutation).
Each budget is one point at (mean_time, mean_best_cost), with
horizontal error = std_time and vertical error = std_cost.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ─────── INPUTS ────────────────────────────────────────────────────────────────
BUDGETS = [
    ("Pop=25, Gen=400, Runs=15",   Path("results"),        "tab:blue",  "o"),
    ("Pop=35, Gen=700, Runs=20",   Path("results_k_4"),    "tab:orange","^"),
    ("Pop=50, Gen=1000, Runs=30",  Path("results_k_4_1000"),"tab:green","s"),
]
CSV_NAME = "E-n22-k4_LNN_mut.csv"
# ────────────────────────────────────────────────────────────────────────────────

def load_metrics(folder: Path) -> pd.DataFrame:
    path = folder / CSV_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path)
    if not {"time", "best_cost"}.issubset(df.columns):
        raise ValueError(f"{CSV_NAME} missing required columns")
    return df

def main():
    fig, ax = plt.subplots(figsize=(6.5, 5))

    for label, folder, color, marker in BUDGETS:
        df = load_metrics(folder)
        t_mean, t_std = df["time"].mean(), df["time"].std()
        c_mean, c_std = df["best_cost"].mean(), df["best_cost"].std()

        ax.errorbar(
            t_mean, c_mean,
            xerr=t_std, yerr=c_std,
            fmt=marker, color=color,
            ecolor=color, elinewidth=2, capsize=5,
            markersize=10, label=label
        )

    ax.set_xlabel("Wall-clock time (s)", fontsize=12)
    ax.set_ylabel("Best cost", fontsize=12)
    ax.set_title("LNN Speed vs. Quality Trade-off (E-n22-k4)", fontsize=14)
    ax.grid(ls="--", alpha=0.4)
    ax.legend(frameon=False, fontsize=10)
    plt.tight_layout()

    out = Path("figures"); out.mkdir(exist_ok=True)
    plt.savefig(out/"speed_quality_pareto.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()