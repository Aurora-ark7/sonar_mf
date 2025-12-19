
from __future__ import annotations
import numpy as np

def complex_lfm_baseband(fs_hz: float, f0_hz: float, f1_hz: float, duration_s: float, taper: str = "hann") -> np.ndarray:
    n = int(round(duration_s * fs_hz))
    t = np.arange(n, dtype=np.float64) / fs_hz
    k = (f1_hz - f0_hz) / duration_s
    phase = 2.0 * np.pi * (f0_hz * t + 0.5 * k * t**2)
    s = np.exp(1j * phase)
    if taper.lower() in ("hann", "hanning"):
        s = s * np.hanning(n)
    elif taper.lower() in ("none", ""):
        pass
    else:
        raise ValueError(f"Unsupported taper: {taper}")
    rms = np.sqrt(np.mean(np.abs(s)**2)) + 1e-12
    return (s / rms).astype(np.complex128)
