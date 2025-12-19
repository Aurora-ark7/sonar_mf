
from __future__ import annotations
import numpy as np
from scipy.signal import fftconvolve

def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()

def fractional_delay_fft(x: np.ndarray, delay_samples: float, nfft: int | None = None) -> np.ndarray:
    x = np.asarray(x)
    n = x.shape[0]
    if nfft is None:
        nfft = next_pow2(n + 2048)
    X = np.fft.fft(x, nfft)
    k = np.arange(nfft)
    phase = np.exp(-1j * 2.0 * np.pi * k * (delay_samples / nfft))
    y = np.fft.ifft(X * phase)[:n]
    return y.astype(np.complex128, copy=False)

def matched_filter_multi(x: np.ndarray, template: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    h = template[::-1].conj()
    M, N = x.shape
    z = np.zeros((M, N), dtype=np.complex128)
    for m in range(M):
        full = fftconvolve(x[m], h, mode="full")
        start = len(h) - 1
        z[m] = full[start:start + N]
    return z
