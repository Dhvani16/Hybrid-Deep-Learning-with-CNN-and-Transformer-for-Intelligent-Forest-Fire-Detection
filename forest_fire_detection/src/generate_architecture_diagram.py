"""
Generates the hybrid CNN+Transformer architecture diagram for the paper.
Output: outputs/plots/architecture_diagram.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "plots", "architecture_diagram.png"
)

# ── colour palette ────────────────────────────────────────────────────────────
C_INPUT      = "#E8F4FD"   # light blue
C_CNN        = "#D4EDDA"   # light green
C_PROJ       = "#FFF3CD"   # light yellow
C_TOKEN      = "#E2D9F3"   # light purple
C_TRANSF     = "#F8D7DA"   # light red/pink
C_CLS        = "#D1ECF1"   # light cyan
C_HEAD       = "#FDECEA"   # light orange
C_OUTPUT     = "#C8E6C9"   # medium green

EDGE = "#333333"
TEXT = "#1a1a1a"

fig, ax = plt.subplots(figsize=(18, 8))
ax.set_xlim(0, 18)
ax.set_ylim(0, 8)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(ax, x, y, w, h, color, label, sublabel=None, fontsize=9, radius=0.25):
    """Draw a rounded rectangle with centred text."""
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        linewidth=1.4, edgecolor=EDGE, facecolor=color, zorder=3,
    )
    ax.add_patch(rect)
    if sublabel:
        ax.text(x, y + 0.18, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=TEXT, zorder=4)
        ax.text(x, y - 0.22, sublabel, ha="center", va="center",
                fontsize=7.5, color="#555555", zorder=4, style="italic")
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=TEXT, zorder=4)


def arrow(ax, x1, y1, x2, y2, label=None):
    """Draw a simple arrow between two points."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.4),
        zorder=2,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.18, label, ha="center", va="bottom",
                fontsize=7.5, color="#444444", zorder=5,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))


def curved_arrow(ax, x1, y1, x2, y2, label=None):
    """Draw a curved/elbow arrow for layout routing."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=EDGE, lw=1.4,
            connectionstyle="arc3,rad=-0.3",
        ),
        zorder=2,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.5
        ax.text(mx, my, label, ha="center", va="bottom",
                fontsize=7.5, color="#444444", zorder=5,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))


# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(9, 7.6, "Hybrid CNN+Transformer Architecture for Forest Fire Detection",
        ha="center", va="center", fontsize=12, fontweight="bold", color=TEXT)

# ── TOP ROW (left → right) ────────────────────────────────────────────────────
# 1. Input Image
box(ax, 1.2, 5.5, 1.6, 1.0, C_INPUT, "Input Image", "224 × 224 × 3")

# 2. EfficientNet-B4 Backbone
box(ax, 3.7, 5.5, 2.6, 1.2, C_CNN, "EfficientNet-B4", "CNN Backbone\n(pretrained ImageNet-1K)", fontsize=9.5)

# 3. CNN Feature Map callout (tensor shape annotation)
box(ax, 6.5, 5.5, 1.6, 0.85, C_CNN, "Feature Map", "B × 1792 × 7 × 7", fontsize=8.5)

# 4. 1×1 Conv + BN Projection
box(ax, 9.0, 5.5, 2.0, 1.0, C_PROJ, "1×1 Conv + BN", "B × 512 × 7 × 7", fontsize=8.5)

# 5. Flatten / Tokenisation
box(ax, 11.6, 5.5, 1.9, 1.0, C_TOKEN, "Flatten + CLS", "50 tokens × 512", fontsize=8.5)

# 6. Positional Embedding (small note above)
ax.text(11.6, 6.25, "+ Positional Embedding", ha="center", va="bottom",
        fontsize=7.5, color="#555555", style="italic")

# Arrows: top row
arrow(ax, 2.01, 5.5, 2.38, 5.5, label="RGB\nimage")
arrow(ax, 4.99, 5.5, 5.68, 5.5, label="conv\nfeatures")
arrow(ax, 7.30, 5.5, 7.98, 5.5, label="spatial\nmap")
arrow(ax, 10.00, 5.5, 10.64, 5.5, label="projected\ntokens")

# ── BOTTOM ROW (right → left) ─────────────────────────────────────────────────
# 7. Transformer Encoder block
box(ax, 9.5, 2.5, 4.0, 1.8, C_TRANSF,
    "Transformer Encoder × 4",
    "MHSA (8 heads) + FFN (d=2048)\npre-LN · dropout=0.1",
    fontsize=9, radius=0.3)

# Small inner detail: 4 stacked layers
for i in range(4):
    mini_y = 2.1 + i * 0.38
    xoff = 9.5
    rect = FancyBboxPatch(
        (xoff + 1.65, mini_y - 0.14), 1.1, 0.27,
        boxstyle="round,pad=0.02", linewidth=0.6,
        edgecolor="#999999", facecolor="#ffffff", alpha=0.5, zorder=5,
    )
    ax.add_patch(rect)
    ax.text(xoff + 2.2, mini_y, f"Layer {i+1}", ha="center", va="center",
            fontsize=6.5, color="#444444", zorder=6)

# 8. CLS Token output
box(ax, 5.0, 2.5, 1.8, 1.0, C_CLS, "CLS Token", "B × 512", fontsize=8.5)

# 9. Classifier Head
box(ax, 2.8, 2.5, 2.2, 1.0, C_HEAD, "Classifier Head", "Dropout(0.3)\n+ Linear(512→2)", fontsize=8.5)

# 10. Output
box(ax, 0.9, 2.5, 1.3, 1.0, C_OUTPUT, "Output", "Fire / No-Fire", fontsize=8.5)

# Arrows: bottom row
arrow(ax, 7.48, 2.5, 5.90, 2.5, label="CLS\nrepresentation")
arrow(ax, 4.09, 2.5, 3.90, 2.5, label="B × 512")
arrow(ax, 1.69, 2.5, 1.55, 2.5)

# ── CONNECTOR from top row → bottom row ──────────────────────────────────────
# From Tokenisation (11.6, 5.0) down to Transformer (11.6, 3.4)
ax.annotate(
    "", xy=(11.6, 3.41), xytext=(11.6, 4.99),
    arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.4),
    zorder=2,
)
ax.text(11.95, 4.2, "token\nsequence", ha="left", va="center",
        fontsize=7.5, color="#444444")

# ── Colour legend ─────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_INPUT,  edgecolor=EDGE, label="Input"),
    mpatches.Patch(facecolor=C_CNN,    edgecolor=EDGE, label="CNN Backbone"),
    mpatches.Patch(facecolor=C_PROJ,   edgecolor=EDGE, label="Projection"),
    mpatches.Patch(facecolor=C_TOKEN,  edgecolor=EDGE, label="Tokenisation"),
    mpatches.Patch(facecolor=C_TRANSF, edgecolor=EDGE, label="Transformer"),
    mpatches.Patch(facecolor=C_CLS,    edgecolor=EDGE, label="CLS Output"),
    mpatches.Patch(facecolor=C_HEAD,   edgecolor=EDGE, label="Classifier"),
    mpatches.Patch(facecolor=C_OUTPUT, edgecolor=EDGE, label="Prediction"),
]
ax.legend(
    handles=legend_items, loc="lower center", ncol=8,
    fontsize=8, frameon=True, framealpha=0.9, edgecolor="#cccccc",
    bbox_to_anchor=(0.5, -0.02),
)

# ── Phase annotations ─────────────────────────────────────────────────────────
# Phase 1 bracket over EfficientNet
ax.annotate(
    "", xy=(2.38, 6.25), xytext=(4.99, 6.25),
    arrowprops=dict(arrowstyle="<->", color="#777777", lw=1.0),
)
ax.text(3.7, 6.45, "Phase 1: frozen backbone", ha="center",
        fontsize=7.5, color="#777777", style="italic")

# Phase 2 bracket over full pipeline
ax.annotate(
    "", xy=(2.38, 6.65), xytext=(12.55, 6.65),
    arrowprops=dict(arrowstyle="<->", color="#777777", lw=1.0),
)
ax.text(7.4, 6.85, "Phase 2: end-to-end fine-tuning (differential LR)", ha="center",
        fontsize=7.5, color="#777777", style="italic")

plt.tight_layout(rect=[0, 0.06, 1, 1])
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUTPUT_PATH}")
