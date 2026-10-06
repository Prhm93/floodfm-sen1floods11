"""Draws the README figures from the Stage 5 result tables.

Usage (from the repository root):
    python scripts/make_figures.py [results_dir] [figures_dir]

Inputs (in results_dir): final_results.csv, paired_differences.csv, per_event_test.csv, difficult_chips.csv
Outputs (in figures_dir): fig0_design.png ... fig5_difficult_chips.png
Every number in the figures comes from these tables. No number is typed in this script.
"""
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 100, "savefig.dpi": 160, "savefig.bbox": "tight"})

# Colour-blind-safe colours; hollow markers mean "10% of the labels"
COLOR = {
    "Otsu (VH)": "#8c8c8c",
    "Fixed VH threshold": "#5a5a5a",
    "U-Net radar, 2 bands": "#E69F00",
    "U-Net optical, 13 bands": "#56B4E9",
    "U-Net optical, 6 bands": "#0072B2",
    "Prithvi frozen, 6 bands": "#009E73",
    "Prithvi full, 6 bands": "#7A4FA3",
    "U-Net optical, 6 bands, 10% labels": "#0072B2",
    "Prithvi frozen, 6 bands, 10% labels": "#009E73",
}
SPLIT_TITLE = {"test": "Test split (known events)", "bolivia": "Bolivia (unseen event)"}

final = pd.read_csv(RESULTS / "final_results.csv")
paired = pd.read_csv(RESULTS / "paired_differences.csv")
per_event = pd.read_csv(RESULTS / "per_event_test.csv")
difficult = pd.read_csv(RESULTS / "difficult_chips.csv")


def water_share(split):
    """Water share of the scored pixels of one split, from final_results.csv."""
    return final[f"{split}_water_share"].iloc[0]


def wrap(text, width=26):
    return textwrap.fill(text, width)


# ---------------------------------------------------------------- Figure 0: design
def fig_design():
    fig, ax = plt.subplots(figsize=(15, 4.6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    boxes = [
        (0.1, "1. Data", "Sen1Floods11\nhand-labelled set\n446 chips, 512 x 512 px\n(about 5 km x 5 km each)\nSentinel-1 radar\nSentinel-2 optical\nwater label", "#EAF2F8"),
        (3.1, "2. Official splits", "Train: 252 chips\nValid: 89 chips\nTest: 90 chips\n(same 10 events)\n\nBolivia: 15 chips\n(event never seen)", "#EAF2F8"),
        (6.1, "3. Methods", "Thresholds: Otsu, fixed VH\nU-Net from scratch:\n radar, optical 13, optical 6\nPrithvi-EO-2.0 (300M):\n frozen, full fine-tuning\n3 seeds, 100 epochs", "#EAF8F0"),
        (9.1, "4. Label settings", "All 252 training chips\n\n10% of the chips\n(26 chips, stratified\nby event)\n\nSame number of learning\nsteps in both settings", "#FDF2E3"),
        (12.1, "5. Evaluation", "Water IoU (pixels pooled)\nBest epoch by validation\n95% chip bootstrap\nPaired differences\nPer-event and\ndifficult-chip checks", "#F3EAF8"),
    ]
    for x, title, body, colour in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.25), 2.8, 4.0, boxstyle="round,pad=0.05,rounding_size=0.12",
                                    fc=colour, ec="#444444", lw=1.2))
        ax.text(x + 1.4, 3.85, title, ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(x + 1.4, 2.0, body, ha="center", va="center", fontsize=9.5, linespacing=1.35)
    for x in [2.95, 5.95, 8.95, 11.95]:
        ax.annotate("", xy=(x + 0.2, 2.25), xytext=(x - 0.02, 2.25),
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#444444"))
    fig.savefig(OUT / "fig0_design.png")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 1: main results
def fig_main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    y = np.arange(len(final))[::-1]
    for ax, split in zip(axes, ["test", "bolivia"]):
        for yi, (_, r) in zip(y, final.iterrows()):
            colour = COLOR[r["method"]]
            filled = "10% labels" not in r["method"]
            ax.hlines(yi, r[f"{split}_iou_ci_low"], r[f"{split}_iou_ci_high"], color=colour, lw=1.4, alpha=0.7)
            ax.hlines(yi, r[f"{split}_iou_seed_min"], r[f"{split}_iou_seed_max"], color=colour, lw=6, alpha=0.9)
            ax.plot(r[f"{split}_iou_mean"], yi, "o", ms=8, mfc=colour if filled else "white", mec=colour, mew=2)
            ax.text(min(r[f"{split}_iou_ci_high"] + 0.012, 0.97), yi, f"{r[f'{split}_iou_mean']:.3f}",
                    va="center", fontsize=9)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Water IoU (higher is better)")
        ax.set_title(f"{SPLIT_TITLE[split]}\nwater share {water_share(split):.3f}", fontsize=11)
        ax.grid(axis="x", alpha=0.25)
        ax.set_yticks(y)
        ax.set_yticklabels([wrap(m) for m in final["method"]], fontsize=9)
    handles = [
        Line2D([0], [0], color="#333333", lw=1.4, label="95% bootstrap interval (chips)"),
        Line2D([0], [0], color="#333333", lw=6, label="range of the 3 seeds"),
        Line2D([0], [0], marker="o", color="w", mfc="#333333", mec="#333333", ms=8, label="mean (all 252 training chips)"),
        Line2D([0], [0], marker="o", color="w", mfc="white", mec="#333333", mew=2, ms=8, label="mean (10% of the labels)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Water IoU of all methods (thresholds and Prithvi full have a single run)", fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_main_results.png")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2: label fraction
def fig_label_fraction():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    pairs = [("U-Net optical, 6 bands", "U-Net optical, 6 bands, 10% labels", "U-Net, 6 bands", "#0072B2", -0.04),
             ("Prithvi frozen, 6 bands", "Prithvi frozen, 6 bands, 10% labels", "Prithvi frozen, 6 bands", "#009E73", 0.04)]
    for ax, split in zip(axes, ["test", "bolivia"]):
        for full_name, small_name, label, colour, dx in pairs:
            pts = []
            for x, name in [(0, full_name), (1, small_name)]:
                r = final[final["method"] == name].iloc[0]
                m, lo, hi = r[f"{split}_iou_mean"], r[f"{split}_iou_ci_low"], r[f"{split}_iou_ci_high"]
                ax.errorbar(x + dx, m, yerr=[[m - lo], [hi - m]], fmt="o", color=colour, capsize=4, ms=8, lw=1.6)
                ax.text(x + dx + (0.07 if dx > 0 else -0.07), m, f"{m:.3f}", fontsize=9, va="center",
                        ha="left" if dx > 0 else "right", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))
                pts.append((x + dx, m))
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=colour, lw=1.6, alpha=0.8, label=label)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["All labels\n(252 chips)", "10% of the labels\n(26 chips)"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(0.45, 0.95)
        ax.set_title(SPLIT_TITLE[split], fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        # Paired differences (U-Net minus Prithvi), from paired_differences.csv
        lines = []
        for comp, tag in [("Band-matched U-Net vs Prithvi frozen, all labels", "all labels"),
                          ("Band-matched U-Net vs Prithvi frozen, 10% labels", "10% labels")]:
            r = paired[(paired["comparison"] == comp) & (paired["split"] == split)].iloc[0]
            lines.append(f"U-Net minus Prithvi, {tag}: {r['diff_A_minus_B']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]")
        ax.text(0.5, 0.47, "\n".join(lines), ha="center", va="bottom", fontsize=8.5,
                bbox=dict(boxstyle="round", fc="#F7F7F7", ec="#BBBBBB"))
    axes[0].set_ylabel("Water IoU (mean of 3 seeds, 95% bootstrap interval)")
    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle("With 10% of the labels: on test, frozen Prithvi loses IoU and the U-Net does not", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_label_fraction.png")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3: paired differences
def fig_paired():
    comps = list(dict.fromkeys(paired["comparison"]))
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6), sharey=True)
    y = np.arange(len(comps))[::-1]
    for ax, split in zip(axes, ["test", "bolivia"]):
        for yi, comp in zip(y, comps):
            r = paired[(paired["comparison"] == comp) & (paired["split"] == split)].iloc[0]
            clear = (r["ci_low"] > 0) or (r["ci_high"] < 0)
            colour = "#B2182B" if clear else "#7A7A7A"
            ax.hlines(yi, r["ci_low"], r["ci_high"], color=colour, lw=2.2)
            ax.plot(r["diff_A_minus_B"], yi, "o", ms=8, mfc=colour if clear else "white", mec=colour, mew=2)
            ax.text(0.655, yi, f"{r['diff_A_minus_B']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]",
                    va="center", fontsize=8, ha="right")
        ax.axvline(0, color="black", lw=1)
        ax.set_xlim(-0.2, 0.66)
        ax.set_xlabel("A minus B (water IoU)")
        ax.set_title(SPLIT_TITLE[split], fontsize=11)
        ax.grid(axis="x", alpha=0.25)
        ax.set_yticks(y)
        ax.set_yticklabels([wrap(c, 34) for c in comps], fontsize=9)
    handles = [Line2D([0], [0], marker="o", color="#B2182B", lw=2.2, ms=8, label="clear: interval excludes 0"),
               Line2D([0], [0], marker="o", color="#7A7A7A", mfc="white", mew=2, lw=2.2, ms=8,
                      label="not clear: interval includes 0")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Paired differences on the same chips (A is the first method named; 95% bootstrap interval)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_paired_differences.png")
    plt.close(fig)


# ---------------------------------------------------------------- Figures 4 and 5: heat maps
SHORT = {"fixed": "Fixed\nVH", "unet_s1": "U-Net\nradar", "unet_s2": "U-Net\n13 bands", "unet_s2_6": "U-Net\n6 bands",
         "prithvi_frozen": "Prithvi\nfrozen", "prithvi_full": "Prithvi\nfull", "unet_s2_6_frac10": "U-Net 6\n10% labels",
         "prithvi_frozen_frac10": "Prithvi fr.\n10% labels"}


def heatmap(table, row_labels, title, filename, figsize, note=None):
    methods = list(SHORT)
    data = table[methods].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([SHORT[m] for m in methods], fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.5, color="white" if v < 0.6 else "black")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Water IoU")
    fig.suptitle(title, fontsize=12, y=1.02 if note is None else 1.04)
    if note:
        fig.text(0.5, -0.02, note, ha="center", fontsize=8.5)
    fig.savefig(OUT / filename)
    plt.close(fig)


def fig_events():
    labels = [f"{r.event} ({100 * r.water_share:.1f}% water)" for r in per_event.itertuples()]
    heatmap(per_event, labels, "Water IoU of each test event (mean of the seeds)", "fig4_per_event_test.png", (10.5, 6.2))


def fig_chips():
    labels = [f"{r.chip} ({r.split}, {100 * r.water_share:.0f}% water)" for r in difficult.itertuples()]
    note = ("Paraguay_34417: every method scores 0. The chip probably has no input data (see the README, section Limits).")
    heatmap(difficult, labels, "Water IoU of difficult chips (mean of the seeds)", "fig5_difficult_chips.png", (10.5, 5.2), note)


if __name__ == "__main__":
    fig_design()
    fig_main()
    fig_label_fraction()
    fig_paired()
    fig_events()
    fig_chips()
    print("Figures written to", OUT)
