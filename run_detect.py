
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np

from sonar_mf.config import load_config
from sonar_mf.waveform import complex_lfm_baseband
from sonar_mf.sim import simulate_pings
from sonar_mf.dsp import matched_filter_multi
from sonar_mf.beamforming import das_beamform_freqdomain
from sonar_mf.cfar import cfar_2d_rayleigh_ca
from sonar_mf.blob import extract_blobs
from sonar_mf.merge import merge_detections_polar
from sonar_mf.viz import save_angle_range_db, save_mask, save_detections_overlay, write_json

def write_csv(rows, path):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/paper_single_ping.yaml")
    ap.add_argument("--outdir", type=str, default="outputs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    s = complex_lfm_baseband(cfg.waveform.fs_hz, cfg.waveform.f0_hz, cfg.waveform.f1_hz, cfg.waveform.duration_s, cfg.waveform.taper)
    hydro = np.array(cfg.hydrophones_xy_m, dtype=float)

    X, gt = simulate_pings(cfg.waveform.fs_hz, cfg.c_mps, hydro, s,
                           cfg.scene.max_range_m, cfg.scene.pri_s, cfg.scene.num_pings,
                           targets=[t.__dict__ for t in cfg.scene.targets],
                           noise_snr_db=cfg.scene.noise_snr_db,
                           seed=cfg.scene.seed,
                           pad_samples=cfg.beamforming.linear_shift_pad)

    angles = np.arange(cfg.beamforming.angles_start, cfg.beamforming.angles_stop, cfg.beamforming.angles_step, dtype=float)
    Ls = int(np.ceil((2.0 * cfg.scene.max_range_m / cfg.c_mps) * cfg.waveform.fs_hz))
    ranges = (np.arange(Ls, dtype=float) / cfg.waveform.fs_hz) * (cfg.c_mps / 2.0)

    z = matched_filter_multi(X[0], s)
    Y = das_beamform_freqdomain(z, cfg.waveform.fs_hz, cfg.c_mps, hydro, angles, cfg.beamforming.linear_shift_pad)[:, :Ls]
    I = np.abs(Y)

    mask = cfar_2d_rayleigh_ca(I, cfg.cfar.pfa, cfg.cfar.guard_a, cfg.cfar.guard_r, cfg.cfar.ref_a, cfg.cfar.ref_r)
    blobs = extract_blobs(mask, I, connectivity=cfg.blob.connectivity, min_area=cfg.blob.min_area)

    dets = []
    for b in blobs:
        ui = int(round(b["cy"])); ri = int(round(b["cx"]))
        if 0 <= ui < len(angles) and 0 <= ri < len(ranges):
            dets.append({"azimuth_deg": float(angles[ui]), "range_m": float(ranges[ri]), "peak": float(b["peak"]), "area": int(b["area"])})
    merged = merge_detections_polar(dets, cfg.merge.psi_r_m, cfg.merge.psi_theta_deg)

    save_angle_range_db(I, angles, ranges, outdir/"angle_range_db.png", dyn_range_db=cfg.viz.dyn_range_db)
    save_mask(mask, angles, ranges, outdir/"cfar_mask.png")
    save_detections_overlay(I, angles, ranges, merged, outdir/"detections_merged.png", dyn_range_db=cfg.viz.dyn_range_db)

    write_json(dets, outdir/"detections_raw.json")
    write_json(merged, outdir/"detections_merged.json")
    write_csv(merged, outdir/"detections_merged.csv")
    print(f"raw={len(dets)}, merged={len(merged)}")

if __name__ == "__main__":
    main()
