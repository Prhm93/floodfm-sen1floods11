"""PyTorch dataset for the Sen1Floods11 hand-labelled chips."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from floodfm.data import read_split, read_chip


class Sen1Floods11(Dataset):
    """One item is one chip: a normalised image (C, H, W) float32 and a label (H, W) int64.

    modality: "s1" (VV, VH) or "s2" (13 bands).
    Normalisation uses per-band mean and std from the training split (stats_path).
    Missing input values (NaN in S1, 0 in S2) are set to 0 after normalisation, which equals the training mean.
    Label values: 1 water, 0 not water, -1 no data (ignored by the loss and the scores).
    augment=True applies random horizontal and vertical flips to the image and the label together.
    """

    def __init__(self, data_root, split, modality, stats_path, augment=False):
        self.data_root = data_root
        self.chip_ids = read_split(data_root, split)
        self.modality = modality
        stats = json.loads(Path(stats_path).read_text())[modality]
        self.mean = np.array(stats["mean"], dtype="float32")[:, None, None]
        self.std = np.array(stats["std"], dtype="float32")[:, None, None]
        self.augment = augment

    def __len__(self):
        return len(self.chip_ids)

    def __getitem__(self, idx):
        chip = read_chip(self.data_root, self.chip_ids[idx])
        image = chip[self.modality]
        missing = ~np.isfinite(image) if self.modality == "s1" else (image == 0)
        image = (image - self.mean) / self.std
        image[missing] = 0.0
        label = chip["label"].astype("int64")

        # Random flips; torch.rand is used so that each data loader worker draws different values
        if self.augment:
            if torch.rand(1).item() < 0.5:
                image, label = image[:, :, ::-1], label[:, ::-1]
            if torch.rand(1).item() < 0.5:
                image, label = image[:, ::-1, :], label[::-1, :]

        return torch.from_numpy(image.copy()), torch.from_numpy(label.copy())
