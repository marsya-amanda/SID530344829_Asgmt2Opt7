"""FFT-based shift-tolerant distances for normalised mel arrays."""
import numpy as np

def aligned_rmse_fft(a: np.ndarray, b: np.ndarray, max_shift: int = 50) -> tuple[float, int]:
    """Return minimum RMSE and circular shift for equal ``(M,T)`` arrays."""
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("expected equal (M,T) arrays")
    n = a.shape[-1]
    if not 0 <= max_shift < n:
        raise ValueError("invalid max_shift")
    fa, fb = np.fft.rfft(a, axis=-1), np.fft.rfft(b, axis=-1)
    corr = np.fft.irfft(np.sum(fa * np.conj(fb), axis=0), n=n)
    shifts = np.arange(-max_shift, max_shift + 1)
    scores = (np.sum(a*a) + np.sum(b*b) - 2* corr[(-shifts) % n]) / a.size
    i = int(np.argmin(scores))
    return float(np.sqrt(max(float(scores[i]), 0.0))), int(shifts[i])

def aligned_rmse(a: np.ndarray, b: np.ndarray, max_shift: int = 50) -> float:
    """Return shift-tolerant RMSE for ``(M,T)`` arrays."""
    return aligned_rmse_fft(a, b, max_shift)[0]