"""U-Net training for Sen1Floods11: model, loss, training loop, validation and checkpoints."""
import json
import random
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss
from torch.utils.data import DataLoader

from floodfm.dataset import Sen1Floods11

IN_CHANNELS = {"s1": 2, "s2": 13, "s2_6": 6}


def set_seed(seed):
    """Fixes the random start of Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(modality):
    """U-Net with a ResNet-34 encoder, random start (no pretrained weights) and 2 output classes."""
    return smp.Unet(encoder_name="resnet34", encoder_weights=None,
                    in_channels=IN_CHANNELS[modality], classes=2)


@torch.no_grad()
def validate(model, loader, device):
    """Water IoU over all scored pixels of the loader; counts are summed before the division."""
    model.eval()
    tp = fp = fn = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = model(images).argmax(dim=1)
        scored = labels != -1
        tp += int(((pred == 1) & (labels == 1) & scored).sum())
        fp += int(((pred == 1) & (labels == 0) & scored).sum())
        fn += int(((pred == 0) & (labels == 1) & scored).sum())
    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0


def train(data_root, stats_path, checkpoint_root, modality, seed,
          epochs=100, batch_size=8, lr=1e-3, weight_decay=1e-4, num_workers=2):
    """Trains one U-Net run. Saves last.pt, best.pt and history.json in checkpoint_root/unet_<modality>_seed<seed>.
    If last.pt exists, training continues from the epoch after the saved one."""
    device = torch.device("cuda")
    run_dir = Path(checkpoint_root) / f"unet_{modality}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    # Data: training chips with random flips; validation chips without flips
    train_ds = Sen1Floods11(data_root, "train", modality, stats_path, augment=True)
    valid_ds = Sen1Floods11(data_root, "valid", modality, stats_path, augment=False)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers, generator=generator)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Model, optimiser, cosine learning-rate schedule, mixed-precision scaler and losses
    model = build_model(modality).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")
    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
    dice_loss = DiceLoss(mode="multiclass", ignore_index=-1)

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
                logits = model(images)
            loss = ce_loss(logits.float(), labels) + dice_loss(logits.float(), labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item()

        # Learning rate used in this epoch, recorded before the schedule moves to the next value
        epoch_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        valid_iou = validate(model, valid_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / len(train_loader),
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

        print(f"epoch {epoch:3d} | loss {history[-1]['train_loss']:.4f} | valid IoU {valid_iou:.4f} "
              f"| best {best_iou:.4f} | {history[-1]['seconds']:.1f} s")

    return history
