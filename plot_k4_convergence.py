#!/usr/bin/env python3
"""
Plot per-generation convergence of six GA operators on E-n22-k4 (no-mutation).
Two vertically-stacked zoom windows:
  • 370–400  (fast-converging operators)
  • 550–700  (slow operators)
Only generations 1–50 are shown.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ─────── CONFIG ──────────────────────────────────────────────────────────────
BASE_DIR = Path("/Users/alparslanguzey/Desktop/GA Crossover Paper-1/results/convergence/E-n22-k4")
OPS      = ["AEX", "BRKGA", "CX", "LNN", "OX", "RK-2PX"]
COLORS   = ["tab:blue", "tab:purple", "tab:orange", "tab:red", "tab:green", "tab:brown"]
MARKERS  = ["o", "D", "s", "X", "^", "v"]
MAX_GEN  = 50                       # plot generations 1 … 50
ZOOMS    = [                        # (ymin, ymax, subtitle) per panel
    (370, 400, "Zoom: 370–400  (fast operators)"),
    (550, 700, "Zoom: 550–700  (slow operators)")
]
# ─────────────────────────────────────────────────────────────────────────────


def load_runs(op_dir: Path) -> list[pd.Series]:
    runs: list[pd.Series] = []
    nomut_dir = op_dir / "nomut"
    if not nomut_dir.exists():
        print(f"⚠️  {nomut_dir} missing — skipping operator.")
        return runs

    for csv in sorted(nomut_dir.glob("run_*_convergence.csv")):
        try:
            df = pd.read_csv(csv)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            print(f"  → skipping {csv.name} (empty / parse error)")
            continue
        if {"generation", "best_cost"}.issubset(df.columns) and not df.empty:
            ser = df.set_index("generation")["best_cost"].iloc[:MAX_GEN]
            ser = ser.reindex(range(1, MAX_GEN + 1)).ffill()
            runs.append(ser)

    if runs:
        print(f"  ✔ {op_dir.name}: loaded {len(runs)} runs")
    else:
        print(f"⚠️  No usable runs in {op_dir}")
    return runs


def summarise(runs: list[pd.Series]) -> pd.DataFrame:
    df = pd.concat(runs, axis=1)
    return pd.DataFrame({
        "mean": df.mean(axis=1),
        "std":  df.std(axis=1)
    })


def main() -> None:
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(7, 8), dpi=120
    )
    fig.suptitle("GA Convergence (E-n22-k4)", fontsize=15)

    for op, color, marker in zip(OPS, COLORS, MARKERS):
        runs = load_runs(BASE_DIR / op)
        if not runs:
            continue
        stats = summarise(runs)

        for ax, (ymin, ymax, subtitle) in zip((ax_top, ax_bot), ZOOMS):
            ax.plot(
                stats.index, stats["mean"],
                label=op,
                color=color,
                marker=marker,
                markevery=5,
                markersize=5,
                linewidth=1.5
            )
            ax.fill_between(
                stats.index,
                stats["mean"] - stats["std"],
                stats["mean"] + stats["std"],
                color=color,
                alpha=0.20
            )
            ax.set_ylim(ymin, ymax)
            ax.set_ylabel("Best cost", fontsize=11)
            ax.set_title(subtitle, fontsize=12, pad=6)
            ax.grid(ls="--", alpha=0.35)

    ax_bot.set_xlabel("Generation", fontsize=11)
    ax_top.set_xlim(1, MAX_GEN)

    # ─── shared legend way below the x-axis ───
    handles, labels = ax_top.get_legend_handles_labels()
    fig.legend(
        handles, labels, title="Operator",
        loc="lower center", ncol=3, fontsize=9, title_fontsize=10,
        frameon=False, bbox_to_anchor=(0.5, -0.005)
    )

    # leave plenty of room at bottom
    plt.tight_layout(rect=[0, 0.12, 1, 0.95])

    out_dir = Path("figures"); out_dir.mkdir(exist_ok=True)
    outfile = out_dir / "k4_convergence_fast_vs_slow.png"
    fig.savefig(outfile, dpi=300)
    print(f"\n✅ Saved plot → {outfile}")

    plt.show()


if __name__ == "__main__":
    main()