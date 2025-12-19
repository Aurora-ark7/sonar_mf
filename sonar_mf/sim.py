
from __future__ import annotations
import numpy as np
from .dsp import fractional_delay_fft, next_pow2

def simulate_pings(fs_hz: float, c_mps: float, hydrophones_xy_m: np.ndarray, waveform: np.ndarray,
                   max_range_m: float, pri_s: float, num_pings: int, targets: list[dict],
                   noise_snr_db: float, seed: int = 0, pad_samples: int = 1024):
    rng = np.random.default_rng(seed)
    hydrophones_xy_m = np.asarray(hydrophones_xy_m, dtype=np.float64)
    M = hydrophones_xy_m.shape[0]

    T_sig = len(waveform) / fs_hz
    t_max = 2.0 * max_range_m / c_mps
    N = int(np.ceil((t_max + T_sig) * fs_hz)) + pad_samples

    nfft_sig = next_pow2(len(waveform) + 2048)

    X = np.zeros((num_pings, M, N), dtype=np.complex128)
    gt = {"pings": []}

    for k in range(num_pings):
        tk = k * pri_s
        x = np.zeros((M, N), dtype=np.complex128)
        gt_ping = {"t_s": tk, "targets": []}

        for tgt in targets:
            pos = np.array([tgt["x0_m"] + tgt["vx_mps"] * tk, tgt["y0_m"] + tgt["vy_mps"] * tk], dtype=np.float64)
            amp = float(tgt.get("amplitude", 1.0))
            tid = int(tgt.get("id", 1))
            gt_ping["targets"].append({"id": tid, "x_m": float(pos[0]), "y_m": float(pos[1])})

            tau_tx = np.linalg.norm(pos) / c_mps
            for m in range(M):
                tau = tau_tx + np.linalg.norm(pos - hydrophones_xy_m[m]) / c_mps
                d = tau * fs_hz
                n_int = int(np.floor(d))
                frac = float(d - n_int)
                w_del = fractional_delay_fft(waveform, frac, nfft_sig)
                start = n_int
                end = start + len(w_del)
                if 0 <= start and end <= N:
                    x[m, start:end] += amp * w_del

        sig_power = float(np.mean(np.abs(x)**2)) + 1e-12
        noise_power = sig_power / (10.0 ** (noise_snr_db / 10.0))
        noise = (rng.normal(0.0, np.sqrt(noise_power/2), size=x.shape) +
                 1j * rng.normal(0.0, np.sqrt(noise_power/2), size=x.shape))
        X[k] = x + noise
        gt["pings"].append(gt_ping)

    return X, gt
