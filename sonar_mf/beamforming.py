
from __future__ import annotations
import numpy as np
from .dsp import next_pow2

def das_beamform_freqdomain(z: np.ndarray, fs_hz: float, c_mps: float, hydrophones_xy_m: np.ndarray,
                            angles_deg: np.ndarray, pad_samples: int = 1024) -> np.ndarray:
    """
    Fractional-delay DAS beamforming (fast):

    - FFT each channel once
    - For each look angle, apply per-channel phase ramp in frequency domain and sum
    - IFFT once per angle

    z: (M, N) complex (per-channel matched filter output)
    Returns Y: (U, N) complex (beamformed time-series per angle)
    """
    z = np.asarray(z, dtype=np.complex128)
    hydrophones_xy_m = np.asarray(hydrophones_xy_m, dtype=np.float64)
    M, N = z.shape
    angles_deg = np.asarray(angles_deg, dtype=np.float64)
    U = len(angles_deg)

    max_ap = np.max(np.linalg.norm(hydrophones_xy_m - hydrophones_xy_m.mean(axis=0), axis=1)) * 2.0
    max_shift = int(np.ceil((max_ap / c_mps) * fs_hz)) + int(pad_samples)
    nfft = next_pow2(N + max_shift + 2048)

    # FFT each channel once
    Zf = np.fft.fft(z, nfft, axis=1)  # (M, nfft)
    k = np.arange(nfft, dtype=np.float64)[None, :]  # (1, nfft)

    Y = np.zeros((U, N), dtype=np.complex128)
    for ui, ang in enumerate(angles_deg):
        phi = np.deg2rad(float(ang))
        uvec = np.array([np.cos(phi), np.sin(phi)], dtype=np.float64)
        d_samples = (hydrophones_xy_m @ uvec) * (fs_hz / c_mps)  # (M,)

        # phase_mk = exp(-j 2pi k d / nfft), build as (M, nfft) via broadcasting
        phase = np.exp(-1j * 2.0 * np.pi * (d_samples[:, None] * k) / nfft)  # (M, nfft)
        Sf = np.sum(Zf * phase, axis=0)  # (nfft,)
        y = np.fft.ifft(Sf, nfft)[:N]
        Y[ui] = y
    return Y
