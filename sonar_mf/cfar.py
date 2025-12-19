
from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter

def cfar_2d_rayleigh_ca(I: np.ndarray, pfa: float, guard_a: int, guard_r: int, ref_a: int, ref_r: int) -> np.ndarray:
    """
    2D CA-CFAR (Rayleigh amplitude, unknown scale), paper-style threshold:
        Tc = sqrt( (Pfa^{-1/Nref} - 1) * sum_{ref} I^2 )

    Implementation is O(U*R) using uniform_filter (integral-image based).
    I: shape (U_angles, R_ranges)
    """
    I = np.asarray(I, dtype=np.float64)
    U, R = I.shape

    wa = int(ref_a + guard_a)
    wr = int(ref_r + guard_r)
    ka = 2 * wa + 1
    kr = 2 * wr + 1

    iga = int(guard_a)
    igr = int(guard_r)
    ia = 2 * iga + 1
    ir = 2 * igr + 1

    Nouter = float(ka * kr)
    Ninner = float(ia * ir)
    Nref = Nouter - Ninner
    if Nref <= 0:
        raise ValueError("Invalid CFAR window: Nref<=0")

    I2 = I * I

    # uniform_filter returns mean over window; multiply by window area to get sum
    sum_outer = uniform_filter(I2, size=(ka, kr), mode="constant", cval=0.0) * Nouter
    sum_inner = uniform_filter(I2, size=(ia, ir), mode="constant", cval=0.0) * Ninner
    sum_ref = np.maximum(sum_outer - sum_inner, 0.0)

    k = (pfa ** (-1.0 / Nref) - 1.0)
    thr = np.sqrt(np.maximum(k * sum_ref, 0.0))

    mask = I > thr
    # Invalidate borders where the reference window is incomplete (matches paper typical handling)
    mask[:wa, :] = False
    mask[U-wa:, :] = False
    mask[:, :wr] = False
    mask[:, R-wr:] = False
    return mask
