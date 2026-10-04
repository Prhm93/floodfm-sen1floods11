"""Prithvi-EO-2.0-300M-TL on Sen1Floods11 through TerraTorch: data, model, training with checkpoints, evaluation.

Settings follow the TerraTorch 1.x Sen1Floods11 example configuration by the Prithvi team:
https://github.com/blumenstiel/TerraTorch-Examples/blob/main/configs/prithvi_v2_eo_300_tl_unet_sen1floods11.yaml
Project rules that apply to all methods: best checkpoint by validation IoU; evaluation on full 512 x 512 chips;
pixels without input are predicted as not water.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from segmentation_models_pytorch.losses import DiceLoss
from terratorch.models import EncoderDecoderFactory
from torch.utils.data import DataLoader, Dataset

from floodfm.data import read_split, read_chip
from floodfm.metrics import confusion_counts, add_counts, scores
from floodfm.train import set_seed

# Prithvi input bands and their positions in the 13-band S2Hand files (B2, B3, B4, B8A, B11, B12)
BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
BAND_INDICES = [1, 2, 3, 8, 11, 12]

# Scale factor and per-band normalisation values from the example configuration (reflectance scale)
CONSTANT_SCALE = 0.0001
MEANS = [0.11076498225107874, 0.13456047562676646, 0.12477149645635542,
         0.3248933937526503, 0.23118412840904512, 0.15624583324071273]
STDS = [0.15469174852002912, 0.13070592427323752, 0.12786689586224442,
        0.13925781946803198, 0.11303782829438778, 0.10207461132314981]

EVAL_SPLITS = ["valid", "test", "bolivia"]


class PrithviSen1Floods11(Dataset):
    """One item is one chip: six Sentinel-2 bands (6, H, W) float32 and the label (H, W) int64.

    Band values are multiplied by 0.0001, NaN values are replaced with 0, then each band is normalised.
    augment=True applies a random D4 transform: one of four rotations by 90 degrees, with or without a flip.
    """

    def __init__(self, data_root, split, augment=False):
        self.data_root = data_root
        self.chip_ids = read_split(data_root, split)
        self.augment = augment
        self.mean = np.array(MEANS, dtype="float32")[:, None, None]
        self.std = np.array(STDS, dtype="float32")[:, None, None]

    def __len__(self):
        return len(self.chip_ids)

    def __getitem__(self, idx):
        chip = read_chip(self.data_root, self.chip_ids[idx])
        image = chip["s2"][BAND_INDICES] * CONSTANT_SCALE
        image = np.nan_to_num(image, nan=0.0)
        image = (image - self.mean) / self.std
        label = chip["label"].astype("int64")

        # Random D4 transform; torch random numbers give different values in each data loader worker
        if self.augment:
            k = int(torch.randint(0, 4, (1,)))
            image = np.rot90(image, k, axes=(1, 2))
            label = np.rot90(label, k, axes=(0, 1))
            if torch.rand(1).item() < 0.5:
                image, label = image[:, :, ::-1], label[:, ::-1]

        return (torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(label)))


def build_prithvi(freeze_backbone):
    """Prithvi-EO-2.0-300M-TL backbone (pretrained, 512 x 512 input) with a UNet decoder, as in the example config.
    freeze_backbone=True keeps the backbone weights fixed; only the necks, decoder and head learn."""
    model = EncoderDecoderFactory().build_model(
        task="segmentation",
        backbone="prithvi_eo_v2_300_tl",
        backbone_pretrained=True,
        backbone_img_size=512,
        backbone_coords_encoding=[],
        backbone_bands=BANDS,
        necks=[{"name": "SelectIndices", "indices": [5, 11, 17, 23]},
               {"name": "ReshapeTokensToImage"},
               {"name": "LearnedInterpolateToPyramidal"}],
        decoder="UNetDecoder",
        decoder_channels=[512, 256, 128, 64],
        head_dropout=0.1,
        num_classes=2,
    )
    if freeze_backbone:
        for p in model.encoder.parameters():
            p.requires_grad = False
    return model


def logits_of(output):
    """Returns the logits tensor from a TerraTorch model output object or a plain tensor."""
    return output.output if hasattr(output, "output") else output


@torch.no_grad()
def validate(model, loader, device, loss_fn):
    """Mean validation loss over batches, and water IoU over all scored pixels (counts summed before the division)."""
    model.eval()
    tp = fp = fn = 0
    loss_sum = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = logits_of(model(images))
        loss_sum += loss_fn(logits.float(), labels).item()
        pred = logits.argmax(dim=1)
        scored = labels != -1
        tp += int(((pred == 1) & (labels == 1) & scored).sum())
        fp += int(((pred == 1) & (labels == 0) & scored).sum())
        fn += int(((pred == 0) & (labels == 1) & scored).sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return loss_sum / len(loader), iou


def train_prithvi(data_root, checkpoint_root, mode, seed, epochs=100, batch_size=8,
                  lr=1e-4, weight_decay=0.1, num_workers=2):
    """Trains one Prithvi run (mode "frozen" or "full"). Saves last.pt, best.pt and history.json in
    checkpoint_root/prithvi_<mode>_seed<seed>. If last.pt exists, training continues from the next epoch."""
    device = torch.device("cuda")
    run_dir = Path(checkpoint_root) / f"prithvi_{mode}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    # Data: training chips with D4 transforms; validation chips without transforms
    train_ds = PrithviSen1Floods11(data_root, "train", augment=True)
    valid_ds = PrithviSen1Floods11(data_root, "valid", augment=False)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers, generator=generator)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Model, optimiser over trainable weights only, plateau schedule on validation loss, scaler, Dice loss
    model = build_prithvi(freeze_backbone=(mode == "frozen")).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = DiceLoss(mode="multiclass", ignore_index=-1)

    # Resume from the last checkpoint when it exists
    start_epoch, best_iou, history = 1, -1.0, []
    last_path = run_dir / "last.pt"
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_iou = state["best_iou"]
        history = state["history"]
        print(f"Resumed after epoch {state['epoch']} (best validation IoU so far: {best_iou:.3f})")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = logits_of(model(images))
            loss = loss_fn(logits.float(), labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item()

        # Learning rate of this epoch, then validation, then the plateau schedule step on the validation loss
        epoch_lr = optimizer.param_groups[0]["lr"]
        valid_loss, valid_iou = validate(model, valid_loader, device, loss_fn)
        scheduler.step(valid_loss)
        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / len(train_loader),
            "valid_loss": valid_loss,
            "valid_iou": valid_iou,
            "lr": epoch_lr,
            "seconds": time.time() - t0,
        })

        # Best checkpoint (model weights only) when the validation IoU improves
        if valid_iou > best_iou:
            best_iou = valid_iou
            torch.save({"model": model.state_dict(), "epoch": epoch, "valid_iou": valid_iou}, run_dir / "best.pt")

        # Last checkpoint (full training state), written under a temporary name and then renamed
        tmp_path = run_dir / "last.pt.tmp"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_iou": best_iou,
            "history": history,
        }, tmp_path)
        tmp_path.replace(last_path)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))

        print(f"epoch {epoch:3d} | loss {history[-1]['train_loss']:.4f} | valid loss {valid_loss:.4f} "
              f"| valid IoU {valid_iou:.4f} | best {best_iou:.4f} | lr {epoch_lr:.1e} | {history[-1]['seconds']:.1f} s")

    return history


@torch.no_grad()
def evaluate_prithvi(data_root, checkpoint_path, mode, method, description,
                     results_path, prediction_path, code_commit):
    """Predicts every chip of valid, test and bolivia with the best checkpoint.
    Pixels where all 13 Sentinel-2 bands are 0 (no image) are predicted as not water (same rule as all methods).
    Saves counts and scores as JSON (results_path) and the Bolivia predictions as compressed NumPy (prediction_path)."""
    device = torch.device("cuda")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_prithvi(freeze_backbone=(mode == "frozen")).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    results = {
        "method": method,
        "description": description,
        "modality": "s2_6bands",
        "checkpoint_epoch": state["epoch"],
        "checkpoint_valid_iou": state["valid_iou"],
        "created": datetime.now().isoformat(timespec="seconds"),
        "code_commit": code_commit,
        "splits": {},
    }
    bolivia_predictions = {}

    for split in EVAL_SPLITS:
        ds = PrithviSen1Floods11(data_root, split, augment=False)
        split_total, event_totals, chips = {}, {}, {}
        for i, chip_id in enumerate(ds.chip_ids):
            image, _ = ds[i]
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = logits_of(model(image.unsqueeze(0).to(device)))
            pred = logits.argmax(dim=1)[0].cpu().numpy().astype("uint8")

            # Missing-input rule: no Sentinel-2 image gives "not water"
            chip = read_chip(data_root, chip_id)
            pred[(chip["s2"] == 0).all(axis=0)] = 0

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
