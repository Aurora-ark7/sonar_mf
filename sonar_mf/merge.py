
from __future__ import annotations
import numpy as np

def angle_diff_deg(a: float, b: float) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)

def merge_detections_polar(dets: list[dict], psi_r_m: float, psi_theta_deg: float) -> list[dict]:
    if not dets:
        return []
    remaining = dets.copy()
    merged = []
    while remaining:
        remaining.sort(key=lambda d: d.get("peak", 0.0), reverse=True)
        seed = remaining.pop(0)
        cluster = [seed]
        changed = True
        while changed:
            changed = False
            new_remaining = []
            for d in remaining:
                close = any(
                    abs(d["range_m"] - c["range_m"]) <= psi_r_m and
                    angle_diff_deg(d["azimuth_deg"], c["azimuth_deg"]) <= psi_theta_deg
                    for c in cluster
                )
                if close:
                    cluster.append(d); changed = True
                else:
                    new_remaining.append(d)
            remaining = new_remaining

        w = np.array([c.get("peak", 1.0) for c in cluster], dtype=float) + 1e-12
        r = float(np.sum([c["range_m"]*wi for c, wi in zip(cluster, w)]) / np.sum(w))
        ang = np.deg2rad(np.array([c["azimuth_deg"] for c in cluster], dtype=float))
        s = float(np.sum(np.sin(ang) * w)); ccos = float(np.sum(np.cos(ang) * w))
        az = float(np.rad2deg(np.arctan2(s, ccos)))
        merged.append({"range_m": r, "azimuth_deg": az, "peak": float(np.max(w)), "count": len(cluster)})
    return merged
