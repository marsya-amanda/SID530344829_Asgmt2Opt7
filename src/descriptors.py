import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from foley.constants import * # type: ignore

BRIGHT, ONSET, DENS = 0, 1, 2
TRANSIENT = {0, 1, 2, 3}          # DogBark, Footstep, GunShot, Keyboard (event-like; cf. [CLN24]'s transient vs ambient split)
MAX_COUNT = 12

def _power(lm): return np.exp(2 * lm)
def frame_db(lm): return 10 * np.log10(_power(lm).sum(0) + 1e-10)

def centroid(lm) -> float:                                     # energy-weighted mean mel-bin index (0..79)
    P = _power(lm); return float((P * np.arange(lm.shape[0])[:, None]).sum() / (P.sum() + 1e-10))

def onset_seconds(lm, drop_db=15.0) -> float:                  # first frame within 15 dB of the peak
    e = frame_db(lm); return int(np.argmax(e >= e.max() - drop_db)) * HOP / SR

def event_count(lm, prom_db=8.0, min_gap_s=0.12) -> int:       # prominent peaks of the smoothed energy envelope
    e = uniform_filter1d(frame_db(lm), 3)
    pk, _ = find_peaks(e, prominence=prom_db, distance=max(1, int(min_gap_s * SR / HOP)))
    return int(min(len(pk), MAX_COUNT))

def shift_right(lm, k):
    floor = np.percentile(lm, 5, axis=1, keepdims=True)
    k = int(np.clip(k, 0, lm.shape[1]))
    return np.concatenate([np.repeat(floor, k, 1), lm[:, : lm.shape[1] - k]], 1)

def compute_all(lm): return dict(centroid=centroid(lm), onset=onset_seconds(lm), count=event_count(lm))

def to_unit(d, class_id, stats) -> np.ndarray:                 # measured -> u in [0,1]^3
    b = np.searchsorted(np.asarray(stats["centroid_q"][class_id]), d["centroid"]) / 100.0
    return np.clip([b, d["onset"] / DURATION, np.log1p(d["count"]) / np.log1p(MAX_COUNT)], 0, 1).astype(np.float32)

def from_user(class_id, stats, brightness=None, onset_s=None, count=None):
    """User values (None = unspecified) -> (u[3], mask[3])."""
    if not 0 <= class_id < N_CLASSES:
        raise ValueError("class_id must be in [0,6]")
    u, m = np.zeros(3, np.float32), np.zeros(3, np.float32)
    if brightness is not None: u[BRIGHT], m[BRIGHT] = np.clip(brightness, 0, 1), 1
    if onset_s is not None:    u[ONSET],  m[ONSET]  = np.clip(onset_s / DURATION, 0, 1), 1
    if count is not None:      u[DENS],   m[DENS]   = np.clip(np.log1p(count) / np.log1p(MAX_COUNT), 0, 1), 1
    return u, m
