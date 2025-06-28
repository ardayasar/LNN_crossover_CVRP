#!/usr/bin/env python3
"""
compute_tstats.py – Paired t‐statistics vs. AEX and SCX baselines
using a single summary_table.csv.

Expects:
  results/summary_table.csv
with columns at least:
  instance, operator, apply_mutation (bool), run, best_cost
"""

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel

# ─────────────────────────────── CONFIG ────────────────────────────────

# Path to your combined summary table:
SUMMARY_CSV = Path("results") / "summary_table.csv"

# Instances and operators to include
INSTANCES = ["E-n22-k4", "E-n51-k5", "E-n76-k10", "E-n101-k14"]
OPERATORS = ["PMX", "OX", "CX", "GX", "HX", "MHX", "SCX", "LNN"]
BASELINES = ["AEX", "SCX"]

# regimes: (apply_mutation flag, LaTeX caption phrase, table label suffix)
REGIMES = [
    (False, "without mutation", "nomut"),
    (True,  "with mutation",    "mut")
]

# ──────────────────────────── LOAD & PREP ─────────────────────────────

df = pd.read_csv(SUMMARY_CSV)

# Ensure correct types
df["apply_mutation"] = df["apply_mutation"].astype(bool)
df["instance"]       = df["instance"].astype(str)
df["operator"]       = df["operator"].astype(str)

def get_costs(inst, op, mut_flag):
    """Return numpy array of best_cost for given instance/operator/regime."""
    sub = df[
        (df["instance"] == inst)
      & (df["operator"] == op)
      & (df["apply_mutation"] == mut_flag)
    ].sort_values("run")
    if sub.empty:
        raise ValueError(f"No data for {inst},{op},mut={mut_flag}")
    return sub["best_cost"].to_numpy()

# ────────────────────────────────── MAIN ──────────────────────────────────

def emit_table(mut_flag, caption, label):
    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(rf"  \caption{{\(t\)-statistics \textbf{{{caption}}}.  "
          r"Positive means the row operator has higher cost than baseline.}}")
    print(rf"  \label{{tab:t_{label}}}")
    print("  \\begin{tabular}{l" + "r" * len(OPERATORS) + "}")
    print("    \\toprule")
    print("    Instance & " + " & ".join(OPERATORS) + r" \\")
    print("    \\midrule")

    # block vs. AEX
    print(r"    \multicolumn{" + f"{len(OPERATORS)+1}" + r"}{c}{\(t\) vs.\ AEX} \\")
    print("    \\midrule")
    for inst in INSTANCES:
        costs_ref = get_costs(inst, "AEX", mut_flag)
        tvals = []
        better = []
        for op in OPERATORS:
            costs_op = get_costs(inst, op, mut_flag)
            t, p = ttest_rel(costs_op, costs_ref)
            tvals .append(f"{t:.2f}")
            # one‐sided: consider "op better" if op_mean < ref_mean and p/2 <0.05
            if costs_op.mean() < costs_ref.mean() and (p/2) < 0.05:
                better.append(r"\textbf{better}")
            else:
                better.append("—")
        print("    " + inst + " & " + " & ".join(tvals) + r" \\")
        print(r"    \quad Better than AEX & " + " & ".join(better) + r" \\")

    # block vs. SCX
    print("    \\midrule")
    print(r"    \multicolumn{" + f"{len(OPERATORS)+1}" + r"}{c}{\(t\) vs.\ SCX} \\")
    print("    \\midrule")
    for inst in INSTANCES:
        costs_ref = get_costs(inst, "SCX", mut_flag)
        tvals = []
        better = []
        for op in OPERATORS:
            costs_op = get_costs(inst, op, mut_flag)
            t, p = ttest_rel(costs_op, costs_ref)
            tvals .append(f"{t:.2f}")
            if costs_op.mean() < costs_ref.mean() and (p/2) < 0.05:
                better.append(r"\textbf{better}")
            else:
                better.append("—")
        print("    " + inst + " & " + " & ".join(tvals) + r" \\")
        print(r"    \quad Better than SCX & " + " & ".join(better) + r" \\")

    print("    \\bottomrule")
    print("  \\end{tabular}")
    print("\\end{table}\n")

def main():
    for mut_flag, caption, label in REGIMES:
        emit_table(mut_flag, caption, label)

if __name__ == "__main__":
    main()