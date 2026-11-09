import numpy as np, librosa, soundfile as sf
from .constants import * # type: ignore

def load_wav(path) -> np.ndarray:
    """Load, resample, mono-mix, and crop/pad a WAV to ``(88200,)`` float32."""
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1: y = y.mean(1)
    if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return np.pad(y, (0, max(0, N_SAMPLES - len(y))))[:N_SAMPLES]

def wav_to_logmel(y: np.ndarray) -> np.ndarray:
    """Convert ``(88200,)`` float audio to a ``(80,345)`` float32 log-magnitude mel."""
    if y.ndim != 1 or len(y) != N_SAMPLES:
        raise ValueError(f"expected mono audio with {N_SAMPLES} samples, got {y.shape}")
    M = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                                       n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=1.0)
    return np.log(np.maximum(M, 1e-5)).astype(np.float32)

def fit_norm(logmels: np.ndarray) -> dict:
    """Fit train-only percentile normalisation statistics from ``(...,80,345)`` mels."""
    lo, hi = np.percentile(logmels, [0.1, 99.9])          # TRAIN logmels only
    return {"lo": float(lo), "hi": float(hi)}

def normalise(lm, lo, hi) -> np.ndarray:
    """Map ``(...,80,345)`` log-mels to ``(...,80,352)`` float32 in ``[-1,1]``."""
    if lm.shape[-2:] != (N_MELS, N_FRAMES):
        raise ValueError(f"expected (...,{N_MELS},{N_FRAMES}), got {lm.shape}")
    if hi <= lo:
        raise ValueError("normalisation hi must be greater than lo")
    x = 2 * (np.clip(lm, lo, hi) - lo) / (hi - lo) - 1
    pad = [(0, 0)] * (x.ndim - 1) + [(0, T_PAD - x.shape[-1])]
    return np.pad(x, pad, constant_values=-1.0).astype(np.float32)

def denormalise(x, lo, hi) -> np.ndarray:
    """Crop ``(...,80,352)`` normalised mels and restore log-mel values."""
    x = np.clip(x[..., :N_FRAMES], -1, 1)
    return (x + 1) / 2 * (hi - lo) + lo

def logmel_to_wav(lm: np.ndarray, seed: int, n_iter: int = 60) -> np.ndarray:
    """Invert a ``(80,345)`` log-mel with seeded Griffin-Lim to ``(88200,)`` float32."""
    if lm.shape != (N_MELS, N_FRAMES):
        raise ValueError(f"expected {(N_MELS, N_FRAMES)} log-mel, got {lm.shape}")
    S = librosa.feature.inverse.mel_to_stft(np.exp(lm), sr=SR, n_fft=N_FFT, power=1.0,
                                            fmin=FMIN, fmax=FMAX)
    y = librosa.griffinlim(S, n_iter=n_iter, hop_length=HOP, win_length=N_FFT, n_fft=N_FFT,
                           length=N_SAMPLES, random_state=seed)       # seeded => reproducible
    return (0.9 * y / (np.abs(y).max() + 1e-8)).astype(np.float32)

def save_wav(path, y, subtype="PCM_16"):
    """Write finite mono ``(88200,)`` audio at the required sample rate."""
    y = np.asarray(y, dtype=np.float32)
    if y.shape != (N_SAMPLES,) or not np.isfinite(y).all():
        raise ValueError("audio must be finite mono float32 with exactly 88200 samples")
    if subtype not in ("PCM_16", "FLOAT"):
        raise ValueError("subtype must be PCM_16 or FLOAT")
    sf.write(path, y, SR, subtype=subtype)
