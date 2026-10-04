"""Pixel counts and scores for binary water maps. Pixels with label -1 are not scored."""


def confusion_counts(pred, label):
    """Returns TP, FP, FN and TN over scored pixels (label 0 or 1). pred: 1 water, 0 not water."""
    scored = label != -1
    p = pred[scored] == 1
    t = label[scored] == 1
    return {
        "tp": int((p & t).sum()),
        "fp": int((p & ~t).sum()),
        "fn": int((~p & t).sum()),
        "tn": int((~p & ~t).sum()),
    }


def add_counts(total, counts):
    """Adds one set of counts to a running total (changed in place) and returns the total."""
    for key in ("tp", "fp", "fn", "tn"):
        total[key] = total.get(key, 0) + counts[key]
    return total


def scores(counts):
    """Returns water IoU, water F1, water share and the number of scored pixels from summed counts."""
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    union = tp + fp + fn
    scored = tp + fp + fn + tn
    return {
        "iou": tp / union if union > 0 else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if union > 0 else None,
        "water_share": (tp + fn) / scored if scored > 0 else None,
        "scored_pixels": scored,
    }
