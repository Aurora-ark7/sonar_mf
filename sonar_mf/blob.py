
from __future__ import annotations
import numpy as np
from scipy.ndimage import label

def extract_blobs(mask: np.ndarray, I: np.ndarray, connectivity: int = 1, min_area: int = 1) -> list[dict]:
    mask = np.asarray(mask, dtype=bool)
    I = np.asarray(I, dtype=np.float64)
    struct = np.ones((3,3), dtype=int) if connectivity == 2 else None
    lab, n = label(mask, structure=struct)
    blobs = []
    for k in range(1, n+1):
        ys, xs = np.where(lab == k)
        area = len(xs)
        if area < min_area:
            continue
        w = I[ys, xs] + 1e-12
        cy = float(np.sum(ys * w) / np.sum(w))
        cx = float(np.sum(xs * w) / np.sum(w))
        peak = float(np.max(I[ys, xs]))
        blobs.append({"cy": cy, "cx": cx, "area": int(area), "peak": peak})
    return blobs
