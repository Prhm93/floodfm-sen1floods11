"""Draws Bolivia example panels: true colour image, true water, and the error map of three methods (seed 0 of each).

STATUS: NOT TESTED. This script was written without access to the data. Run it once on a CPU server and check the output.

Needs: the Sen1Floods11 files and the Bolivia prediction files saved by the evaluation code
(content/predictions/<run name>_bolivia.npz, one array of 512 x 512 values for each chip ID).

Usage:
    python scripts/make_bolivia_panels.py <content_dir> <figures_dir> [chip_id ...]
Example:
    python scripts/make_bolivia_panels.py /content/drive/MyDrive/2026Research/FloodFM-Sen1Floods11/content figures
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

CONTENT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
CHIPS = sys.argv[3:] or ["Bolivia_129334", "Bolivia_312675", "Bolivia_242570"]

HAND = CONTENT / "data" / "sen1floods11" / "HandLabeled"
PRED = CONTENT / "predictions"
MODELS = [("U-Net radar", "unet_s1_seed0"), ("U-Net 6 bands", "unet_s2_6_seed0"), ("Prithvi frozen", "prithvi_frozen_seed0")]

# Error classes: 0 no data, 1 correct land, 2 correct water, 3 false alarm, 4 missed water
ERROR_CMAP = ListedColormap(["dimgrey", "white", "forestgreen", "red", "orange"])
LABEL_CMAP = ListedColormap(["dimgrey", "white", "royalblue"])


def read(layer, chip):
    """Reads one GeoTIFF layer of a chip as an array (bands, height, width)."""
    with rasterio.open(HAND / layer / f"{chip}_{layer}.tif") as src:
        return src.read()


def true_colour(s2):
    """True colour image from Sentinel-2 bands B4, B3, B2 (indices 3, 2, 1); display stretch only."""
    rgb = np.dstack([s2[i].astype("float32") for i in (3, 2, 1)])
    rgb[rgb == 0] = np.nan
    out = np.zeros_like(rgb)
    for k in range(3):
        lo, hi = np.nanpercentile(rgb[..., k], [2, 98])
        out[..., k] = np.clip((rgb[..., k] - lo) / (hi - lo), 0, 1)
    return np.nan_to_num(out, nan=0.4)


def error_map(pred, label):
    """Error classes of one prediction against the label; pixels with label -1 are not scored."""
    scored = label != -1
    err = np.zeros(label.shape, dtype="uint8")
    err[scored & (pred == 0) & (label == 0)] = 1
    err[scored & (pred == 1) & (label == 1)] = 2
    err[scored & (pred == 1) & (label == 0)] = 3
    err[scored & (pred == 0) & (label == 1)] = 4
    return err


def chip_iou(err):
    """Water IoU of one chip from its error map."""
    tp, fp, fn = (err == 2).sum(), (err == 3).sum(), (err == 4).sum()
    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")


predictions = {run: np.load(PRED / f"{run}_bolivia.npz") for _, run in MODELS}

fig, axes = plt.subplots(len(CHIPS), 2 + len(MODELS), figsize=(3.1 * (2 + len(MODELS)), 3.2 * len(CHIPS)), squeeze=False)
for row, chip in enumerate(CHIPS):
    label = read("LabelHand", chip)[0]
    axes[row, 0].imshow(true_colour(read("S2Hand", chip)))
    axes[row, 0].set_title(f"{chip}\nSentinel-2 true colour", fontsize=9)
    axes[row, 1].imshow(label, cmap=LABEL_CMAP, vmin=-1, vmax=1, interpolation="nearest")
    axes[row, 1].set_title("Label (blue = water)", fontsize=9)
    for col, (name, run) in enumerate(MODELS, start=2):
        err = error_map(predictions[run][chip], label)
        axes[row, col].imshow(err, cmap=ERROR_CMAP, vmin=0, vmax=4, interpolation="nearest")
        axes[row, col].set_title(f"{name}\nchip IoU {chip_iou(err):.3f}", fontsize=9)
for ax in axes.ravel():
    ax.axis("off")
fig.legend(handles=[Patch(color="forestgreen", label="correct water"), Patch(color="red", label="false alarm"),
                    Patch(color="orange", label="missed water"), Patch(facecolor="white", edgecolor="black", label="correct land"),
                    Patch(color="dimgrey", label="no data")],
           loc="lower center", ncol=5, frameon=False)
fig.suptitle("Bolivia examples (seed 0 of each method)", fontsize=12)
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
fig.savefig(OUT / "fig6_bolivia_examples.png", dpi=160, bbox_inches="tight")
print("Saved", OUT / "fig6_bolivia_examples.png")
