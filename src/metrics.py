import numpy as np
import soundfile as sf
import torch
from foley.constants import N_CLASSES, N_MELS, N_SAMPLES, SR, T_PAD


def check_wav_format(path) -> dict:
    """Return format and signal validity flags for a WAV path."""
    info = sf.info(path)
    y, _ = sf.read(path, dtype="float32")
    return {"len_ok": info.frames == N_SAMPLES, "sr_ok": info.samplerate == SR,
            "mono_ok": info.channels == 1, "fmt_ok": info.subtype in ("PCM_16", "FLOAT"),
            "finite": bool(np.isfinite(y).all()),
            "nonsilent": float(np.sqrt((y ** 2).mean())) > 1e-3}


@torch.no_grad()
def classify(clf, X, dev, bs=64):
    """Classify ``(N,80,352)`` mels and return probabilities and embeddings."""
    X = np.asarray(X)
    if X.ndim != 3 or X.shape[1:] != (N_MELS, T_PAD):
        raise ValueError(f"expected (N,{N_MELS},{T_PAD}), got {X.shape}")
    clf.eval()
    probs, emb = [], []
    for i in range(0, len(X), bs):
        logits, values = clf(torch.as_tensor(X[i:i + bs], dtype=torch.float32, device=dev)[:, None])
        probs.append(logits.softmax(-1).cpu().numpy())
        emb.append(values.cpu().numpy())
    return np.concatenate(probs), np.concatenate(emb)


def accuracy_confusion(probs, labels):
    """Compute accuracy, per-class accuracy, and a ``(7,7)`` confusion matrix."""
    pred, labels = np.asarray(probs).argmax(-1), np.asarray(labels, dtype=int)
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for truth, guess in zip(labels, pred):
        cm[truth, guess] += 1
    per = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(N_CLASSES, float), where=cm.sum(1) > 0)
    return float((pred == labels).mean()), per, cm


def pairwise_rmse(X) -> float:
    """Return mean pairwise RMSE over ``X=(N,80,352)``."""
    X = np.asarray(X, dtype=np.float32)
    if len(X) < 2:
        return 0.0
    return float(np.mean([np.sqrt(np.mean((X[i] - X[j]) ** 2))
                          for i in range(len(X)) for j in range(i)]))


def nn_train(emb_q, emb_train):
    """Return nearest training indices and cosine similarities."""
    q, tr = np.asarray(emb_q), np.asarray(emb_train)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    tr = tr / np.maximum(np.linalg.norm(tr, axis=1, keepdims=True), 1e-12)
    sims = q @ tr.T
    idx = sims.argmax(1)
    return idx, sims[np.arange(len(q)), idx]


def aligned_rmse(a, b, max_shift=50) -> float:
    """Return minimum time-shifted RMSE for equally shaped arrays."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("aligned_rmse inputs must have equal shapes")
    return min(float(np.sqrt(((a - np.roll(b, s, axis=-1)) ** 2).mean()))
               for s in range(-max_shift, max_shift + 1))
