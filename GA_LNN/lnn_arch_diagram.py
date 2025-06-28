#!/usr/bin/env python3
# lnn_arch_diagram.py
"""
Draw the LNN-Crossover architecture:
Input (N×3) → Linear(3→48) → LiquidLayer ×2 → Head (48 → 1)
→ Softsort(τ) → Argmax → Child Tour.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib import transforms

# ───────────────────────────── CONFIG ──────────────────────────────
BOX_W, BOX_H = 2.6, 0.9        # generic width / height for blocks
H_GAP        = 1.7             # vertical spacing between blocks
FONT_SIZE    = 9

# Output directory
OUT_DIR  = Path("figures")
OUT_DIR.mkdir(exist_ok=True)
OUT_PNG = OUT_DIR / "lnn_arch_python.png"

# ────────────────────────── helper functions ───────────────────────
def add_box(ax, center, text, **kw):
    """Draw a rectangle centered at `center` with label `text`."""
    x, y   = center
    box    = Rectangle((x - BOX_W / 2, y - BOX_H / 2),
                       BOX_W, BOX_H,
                       linewidth=1.2,
                       edgecolor="black",
                       facecolor="white",
                       zorder=2,
                       **kw)
    ax.add_patch(box)
    ax.text(x, y, text,
            ha="center", va="center",
            fontsize=FONT_SIZE,
            wrap=True)

def add_arrow(ax, start, end, **kw):
    """Arrow from start(x,y) to end(x,y)."""
    ax.add_patch(
        FancyArrowPatch(start, end,
                        arrowstyle="->",
                        linewidth=1.1,
                        mutation_scale=8,
                        color="black",
                        zorder=3,
                        **kw)
    )

# ────────────────────────────── main ───────────────────────────────
def main() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 7), dpi=200)
    ax.axis("off")  # no axes

    # y-coordinates (top → bottom)
    y0 = 0
    y1 = y0 - H_GAP
    y2 = y1 - H_GAP
    y3 = y2 - H_GAP
    y4 = y3 - H_GAP
    y5 = y4 - H_GAP

    # Boxes
    add_box(ax, (0, y0),
            "Input\n$N \\times 3$")

    add_box(ax, (0, y1),
            "Linear proj.\n$3 \\;\\rightarrow\\; 48$")

    add_box(ax, (0, y2),
            "Liquid Layer 1\n(48, steps = 1)")

    add_box(ax, (0, y3),
            "Liquid Layer 2\n(48, steps = 1)")

    add_box(ax, (0, y4),
            "MLP head\n48 → 48 → 1")

    add_box(ax, (0, y5),
            "Softsort $(\\tau)$\n+ Argmax\n→ child tour")

    # Arrows
    add_arrow(ax, (0, y0 - BOX_H / 2), (0, y1 + BOX_H / 2))
    add_arrow(ax, (0, y1 - BOX_H / 2), (0, y2 + BOX_H / 2))
    add_arrow(ax, (0, y2 - BOX_H / 2), (0, y3 + BOX_H / 2))
    add_arrow(ax, (0, y3 - BOX_H / 2), (0, y4 + BOX_H / 2))
    add_arrow(ax, (0, y4 - BOX_H / 2), (0, y5 + BOX_H / 2))

    # Tidy limits
    ax.set_xlim(-3, 3)
    ax.set_ylim(y5 - 1.2, 1)

    plt.tight_layout()
    fig.savefig(OUT_PNG, transparent=True)
    print(f"✅ Saved architecture diagram → {OUT_PNG}")

if __name__ == "__main__":
    main()