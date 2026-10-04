"""Baselines that do not learn from the training data."""
import numpy as np
from skimage.filters import threshold_otsu


def otsu_vh(s1):
    """Otsu threshold on the VH band (index 1) of one chip.
    Returns the prediction (H, W; 1 water, 0 not water) and the threshold in dB.
    Pixels without radar data are predicted as not water."""
    vh = s1[1]
    has_radar = np.isfinite(vh)
    pred = np.zeros(vh.shape, dtype="uint8")
    if has_radar.sum() < 2:
        return pred, None
    threshold = float(threshold_otsu(vh[has_radar]))
    pred[has_radar & (vh < threshold)] = 1
    return pred, threshold
