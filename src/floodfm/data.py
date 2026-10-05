"""Data access for the Sen1Floods11 hand-labelled set: official splits, chip reading and label-fraction subsets."""
from pathlib import Path

import numpy as np
import rasterio

# Official split names and the CSV file that lists the chips of each split
SPLIT_FILES = {
    "train": "flood_train_data.csv",
    "valid": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "bolivia": "flood_bolivia_data.csv",
}


def read_split(data_root, split):
    """Returns the chip IDs of one official split in file order, e.g. "Ghana_103272"."""
    csv_path = Path(data_root) / "splits" / "flood_handlabeled" / SPLIT_FILES[split]
    lines = [line for line in csv_path.read_text().splitlines() if line.strip()]
    return [line.split(",")[0].replace("_S1Hand.tif", "") for line in lines]


def read_chip(data_root, chip_id):
    """Returns one chip as a dict: s1 (2, H, W) float32 in dB, s2 (13, H, W) float32, label (H, W) int16."""
    hand_dir = Path(data_root) / "HandLabeled"
    arrays = {}
    for layer in ["S1Hand", "S2Hand", "LabelHand"]:
        with rasterio.open(hand_dir / layer / f"{chip_id}_{layer}.tif") as src:
            arrays[layer] = src.read()
    return {
        "s1": arrays["S1Hand"].astype("float32"),
        "s2": arrays["S2Hand"].astype("float32"),
        "label": arrays["LabelHand"][0].astype("int16"),
    }


def label_fraction_subset(data_root, fraction, subset_seed=0):
    """Fixed subset of the official training chips, stratified by flood event.

    For each event, about `fraction` of its training chips are kept (at least one chip per event).
    The same fraction and subset_seed always return the same chips, in a stable order."""
    rng = np.random.default_rng(subset_seed)
    by_event = {}
    for chip_id in read_split(data_root, "train"):
        by_event.setdefault(chip_id.split("_")[0], []).append(chip_id)

    subset = []
    for event in sorted(by_event):
        ids = sorted(by_event[event])
        n_keep = max(1, int(round(fraction * len(ids))))
        chosen = sorted(rng.permutation(len(ids))[:n_keep])
        subset.extend(ids[i] for i in chosen)
    return subset
