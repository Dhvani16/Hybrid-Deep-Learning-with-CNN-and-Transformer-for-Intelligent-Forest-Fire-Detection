"""
Hybrid CNN+Transformer Architecture Diagram.
Layout: top row left→right, right-side L-connector, bottom row right→left.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "plots", "architecture_diagram.png"
)

C = dict(
    input="#DAEAF8", cnn="#C8E6C9", proj="#FFF8DC", token="#E8DAEF",
    trans="#FADBD8", cls="#D6EAF8", head="#FDEBD0", out="#A9DFBF",
    edge="#2C3E50", txt="#1A1A1A", gray="#7F8C8D",
)

fig, ax = plt.subplots(figsize=(16, 5.8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 5.8)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(cx, cy, w, h, fc, title, sub=None, fs=9.2):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.06",
        lw=1.3, edgecolor=C["edge"], facecolor=fc, zorder=3))
    if sub:
        ax.text(cx, cy + 0.17, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=C["txt"], zorder=4)
        ax.text(cx, cy - 0.18, sub, ha="center", va="center",
                fontsize=6.7, color="#444", style="italic", zorder=4)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=C["txt"], zorder=4)


def harrow(x1, x2, y, label=None):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=C["edge"], lw=1.25), zorder=2)
    if label:
        ax.text((x1 + x2) / 2, y + 0.17, label, ha="center", va="bottom",
                fontsize=6.6, color="#555",
                bbox=dict(boxstyle="round,pad=0.06", fc="white", ec="none", alpha=0.85))


# ─── Title ────────────────────────────────────────────────────────────────────
ax.text(8.0, 5.55,
        "Hybrid CNN+Transformer Architecture for Forest Fire Detection",
        ha="center", va="center", fontsize=12, fontweight="bold", color=C["txt"])

# ─── Phase spans ──────────────────────────────────────────────────────────────
Y1, Y2 = 3.90, 1.88

ax.annotate("", xy=(2.22, 4.62), xytext=(5.02, 4.62),
            arrowprops=dict(arrowstyle="<->", color=C["gray"], lw=0.9))
ax.text(3.62, 4.73, "Phase 1 — frozen CNN", ha="center",
        fontsize=7.0, color=C["gray"], style="italic")

ax.annotate("", xy=(2.22, 4.88), xytext=(10.62, 4.88),
            arrowprops=dict(arrowstyle="<->", color=C["gray"], lw=0.9))
ax.text(6.42, 4.99, "Phase 2 — end-to-end fine-tuning (differential LR)",
        ha="center", fontsize=7.0, color=C["gray"], style="italic")

# ─── Top row (left → right) ───────────────────────────────────────────────────
#  Box edges:  Input [0.4, 2.0]  EfficientNet [2.2, 5.0]  Conv [5.5, 7.5]  Flatten [8.4, 10.6]
box(1.20, Y1, 1.60, 0.90, C["input"], "Input Image",     "224×224×3")
box(3.60, Y1, 2.80, 1.10, C["cnn"],   "EfficientNet-B4", "CNN Backbone (ImageNet-1K)", fs=9.5)
box(6.50, Y1, 2.00, 0.90, C["proj"],  "1×1 Conv + BN",   "B×512×7×7")
box(9.50, Y1, 2.20, 0.90, C["token"], "Flatten + CLS",   "50 tokens × 512")
ax.text(9.50, 4.51, "+ Positional Embedding", ha="center",
        fontsize=6.8, color="#5D4E8A", style="italic")

harrow(2.02, 2.18, Y1, "RGB")
harrow(5.02, 5.48, Y1, "7×7×1792")
harrow(7.52, 8.38, Y1, "7×7×512")

# ─── L-connector (exits top-row Flatten right → goes right → down → enters Transformer right) ──
# Flatten right = 10.6, Transformer right = 12.5
ax.plot([10.62, 12.8, 12.8], [Y1, Y1, Y2],
        color=C["edge"], lw=1.4, solid_capstyle="round", zorder=2)
harrow(12.80, 12.52, Y2)
ax.text(13.00, (Y1 + Y2) / 2, "token\nsequence", ha="left", va="center",
        fontsize=7.0, color="#555")

# ─── Bottom row (data flows right→left through diagram) ───────────────────────
#  Transformer (large, rightmost)  cx=10.0  w=5.0  → edges [7.5, 12.5]
box(10.00, Y2, 5.00, 1.58, C["trans"],
    "Transformer Encoder × 4",
    "MHSA (8 heads, d=512)  +  FFN (d=2048)  ·  pre-LN  ·  dropout=0.1",
    fs=9.5)

# layer chips — centred inside Transformer box (cx=10.0, w=5.0 → inner [8.2, 11.8])
for i, lbl in enumerate(["L1", "L2", "L3", "L4"]):
    lx = 8.55 + i * 1.08  # 8.55, 9.63, 10.71, 11.79 — all inside [7.5, 12.5]
    ax.add_patch(FancyBboxPatch(
        (lx - 0.40, Y2 - 0.54), 0.80, 0.26,
        boxstyle="round,pad=0.02", lw=0.5,
        edgecolor="#AAA", facecolor="white", alpha=0.65, zorder=5))
    ax.text(lx, Y2 - 0.41, lbl, ha="center", va="center",
            fontsize=7.0, color="#555", zorder=6)

#  CLS Token   cx=5.7  w=2.0  → edges [4.7, 6.7]   gap to Transformer left (7.5) = 0.8
box(5.70, Y2, 2.00, 0.92, C["cls"],  "CLS Token",       "B × 512")
#  Classifier  cx=3.20 w=2.40 → edges [2.0, 4.4]   gap to CLS left (4.7) = 0.3
box(3.20, Y2, 2.40, 0.96, C["head"], "Classifier Head", "Dropout(0.3)\n+ Linear(512→2)")
#  Output      cx=1.00 w=1.50 → edges [0.25, 1.75]  gap to Classifier left (2.0) = 0.25
box(1.00, Y2, 1.50, 0.90, C["out"],  "Output",          "Fire / No-Fire")

# arrows bottom row (pointing left)
harrow(7.48, 6.72, Y2)
ax.text(7.10, Y2 + 0.32, "CLS repr.", ha="center", fontsize=7.0, color="#555")

harrow(4.68, 4.42, Y2)
ax.text(4.55, Y2 + 0.32, "B×512", ha="center", fontsize=7.0, color="#555")

harrow(1.98, 1.77, Y2)

# ─── Legend ───────────────────────────────────────────────────────────────────
ax.legend(handles=[
    mpatches.Patch(facecolor=C["input"], edgecolor=C["edge"], label="Input"),
    mpatches.Patch(facecolor=C["cnn"],   edgecolor=C["edge"], label="CNN Backbone"),
    mpatches.Patch(facecolor=C["proj"],  edgecolor=C["edge"], label="Projection"),
    mpatches.Patch(facecolor=C["token"], edgecolor=C["edge"], label="Tokenisation"),
    mpatches.Patch(facecolor=C["trans"], edgecolor=C["edge"], label="Transformer"),
    mpatches.Patch(facecolor=C["cls"],   edgecolor=C["edge"], label="CLS Output"),
    mpatches.Patch(facecolor=C["head"],  edgecolor=C["edge"], label="Classifier"),
    mpatches.Patch(facecolor=C["out"],   edgecolor=C["edge"], label="Prediction"),
], loc="lower center", ncol=8, fontsize=7.5,
   frameon=True, framealpha=0.92, edgecolor="#CCC",
   bbox_to_anchor=(0.5, 0.005))

plt.tight_layout(rect=[0, 0.07, 1, 0.97])
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print("Saved:", OUTPUT_PATH)
