"""Evaluation of a trained U-Net: predictions with the missing-input rule and a results file in the baseline format."""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from floodfm.data import read_chip
from floodfm.dataset import Sen1Floods11
from floodfm.metrics import confusion_counts, add_counts, scores
from floodfm.train import build_model

EVAL_SPLITS = ["valid", "test", "bolivia"]


def missing_input_mask(chip, modality):
    """True where the input has no data: NaN in VH for s1; all 13 bands equal to 0 for s2."""
    if modality == "s1":
        return ~np.isfinite(chip["s1"][1])
    return (chip["s2"] == 0).all(axis=0)


@torch.no_grad()
def evaluate_run(data_root, stats_path, checkpoint_path, modality, method, description,
                 results_path, prediction_path, code_commit):
    """Predicts every chip of valid, test and bolivia with the best checkpoint.
    Pixels without input are predicted as not water (same rule as the baselines).
    Saves counts and scores as JSON (results_path) and the Bolivia predictions as compressed NumPy (prediction_path)."""
    device = torch.device("cuda")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(modality).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    results = {
        "method": method,
        "description": description,
        "modality": modality,
        "checkpoint_epoch": state["epoch"],
        "checkpoint_valid_iou": state["valid_iou"],
        "created": datetime.now().isoformat(timespec="seconds"),
        "code_commit": code_commit,
        "splits": {},
    }
    bolivia_predictions = {}

    for split in EVAL_SPLITS:
        ds = Sen1Floods11(data_root, split, modality, stats_path, augment=False)
        split_total, event_totals, chips = {}, {}, {}
        for i, chip_id in enumerate(ds.chip_ids):
            image, _ = ds[i]
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(image.unsqueeze(0).to(device))
            pred = logits.argmax(dim=1)[0].cpu().numpy().astype("uint8")

            # Missing-input rule: no input data gives "not water"
            chip = read_chip(data_root, chip_id)
            pred[missing_input_mask(chip, modality)] = 0

            counts = confusion_counts(pred, chip["label"])
            chips[chip_id] = counts
            add_counts(split_total, counts)
            add_counts(event_totals.setdefault(chip_id.split("_")[0], {}), counts)
            if split == "bolivia":
                bolivia_predictions[chip_id] = pred

        results["splits"][split] = {
            "overall": {**split_total, **scores(split_total)},
            "events": {event: {**c, **scores(c)} for event, c in sorted(event_totals.items())},
            "chips": chips,
        }

    Path(results_path).write_text(json.dumps(results, indent=2))
    np.savez_compressed(prediction_path, **bolivia_predictions)
    return results
