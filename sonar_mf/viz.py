
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def write_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")

def save_angle_range_db(I_mag, angles_deg, ranges_m, out_path, dyn_range_db=50.0, title="Angle-Range (dB)"):
    I = np.asarray(I_mag, dtype=float)
    I = I / (np.max(I) + 1e-12)
    I_db = 20*np.log10(I + 1e-12)
    I_db = np.clip(I_db, -dyn_range_db, 0.0)
    plt.figure()
    extent = [angles_deg[0], angles_deg[-1], ranges_m[0], ranges_m[-1]]
    plt.imshow(I_db.T, aspect="auto", origin="lower", extent=extent)
    plt.xlabel("Azimuth (deg)"); plt.ylabel("Range (m)")
    plt.title(f"{title} (dyn={dyn_range_db:.0f} dB)")
    plt.colorbar(label="dB (norm)")
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def save_mask(mask, angles_deg, ranges_m, out_path, title="2D CA-CFAR mask"):
    plt.figure()
    extent = [angles_deg[0], angles_deg[-1], ranges_m[0], ranges_m[-1]]
    plt.imshow(mask.astype(float).T, aspect="auto", origin="lower", extent=extent, vmin=0, vmax=1)
    plt.xlabel("Azimuth (deg)"); plt.ylabel("Range (m)")
    plt.title(title); plt.colorbar(label="Mask")
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def save_detections_overlay(I_mag, angles_deg, ranges_m, dets, out_path, dyn_range_db=50.0, title="Angle-Range with detections"):
    I = np.asarray(I_mag, dtype=float)
    I = I / (np.max(I) + 1e-12)
    I_db = 20*np.log10(I + 1e-12)
    I_db = np.clip(I_db, -dyn_range_db, 0.0)
    plt.figure()
    extent = [angles_deg[0], angles_deg[-1], ranges_m[0], ranges_m[-1]]
    plt.imshow(I_db.T, aspect="auto", origin="lower", extent=extent)
    if dets:
        plt.scatter([d["azimuth_deg"] for d in dets], [d["range_m"] for d in dets], s=18)
    plt.xlabel("Azimuth (deg)"); plt.ylabel("Range (m)")
    plt.title(title); plt.colorbar(label="dB (norm)")
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def save_tracks_xy(tracks_hist, gt, out_path):
    plt.figure()
    gt_by_id = {}
    for ping in gt.get("pings", []):
        for tgt in ping.get("targets", []):
            tid = tgt["id"]
            gt_by_id.setdefault(tid, {"x": [], "y": []})
            gt_by_id[tid]["x"].append(tgt["x_m"]); gt_by_id[tid]["y"].append(tgt["y_m"])
    for tid, xy in gt_by_id.items():
        plt.plot(xy["x"], xy["y"], linestyle="--", label=f"GT {tid}")

    any_est = False
    for tr in tracks_hist:
        xs = [h["x"][0] for h in tr["history"]]
        ys = [h["x"][1] for h in tr["history"]]
        if len(xs) >= 2:
            any_est = True
            plt.plot(xs, ys, label=f"Track {tr['id']}{' (C)' if tr.get('confirmed') else ''}")

    plt.axis("equal"); plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.title("Tracks in XY (est vs GT)")
    if any_est or gt_by_id:
        plt.legend(loc="best", fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def save_track_speed_curve(tracks_hist, out_path):
    plt.figure()
    any_line = False
    for tr in tracks_hist:
        ts = [h["t_s"] for h in tr["history"]]
        v = [float(np.hypot(h["x"][2], h["x"][3])) for h in tr["history"]]
        if len(ts) >= 2:
            any_line = True
            plt.plot(ts, v, label=f"Track {tr['id']}")
    plt.xlabel("t (s)"); plt.ylabel("speed (m/s)")
    plt.title("Track speed vs time")
    if any_line:
        plt.legend(loc="best", fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()
