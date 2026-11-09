import subprocess, sys, numpy as np, soundfile as sf, pytest
import torch
from scripts.train_classifier import preloaded_batches
W = "weights/core.pt"
def run(cid, seed, out): subprocess.run([sys.executable, "infer.py", "--weights", W, "--class-id", str(cid),
                                         "--seed", str(seed), "--out", str(out)], check=True)
@pytest.mark.parametrize("cid", range(7))
def test_format(tmp_path, cid):
    o = tmp_path / "a.wav"; run(cid, 0, o); y, sr = sf.read(o); i = sf.info(o)
    assert (i.frames, sr, i.channels) == (88200, 22050, 1) and i.subtype in ("PCM_16", "FLOAT")
    assert np.isfinite(y).all() and np.sqrt((y ** 2).mean()) > 1e-3
def test_seed(tmp_path):
    a, b, c = (tmp_path / f"{n}.wav" for n in "abc"); run(1, 0, a); run(1, 0, b); run(1, 1, c)
    ya, yb, yc = (sf.read(p)[0] for p in (a, b, c))
    assert np.allclose(ya, yb, atol=1e-4) and np.sqrt(((ya - yc) ** 2).mean()) > 1e-3

def test_preloaded_batches_are_complete_and_deterministic():
    x = torch.arange(5 * 80 * 352, dtype=torch.float16).reshape(5, 80, 352)
    y = torch.arange(5, dtype=torch.long)
    first = list(preloaded_batches(x, y, 2, seed=7, augment=False))
    second = list(preloaded_batches(x, y, 2, seed=7, augment=False))
    assert [batch[0].shape for batch in first] == [(2, 1, 80, 352), (2, 1, 80, 352), (1, 1, 80, 352)]
    assert all(batch[0].dtype == torch.float32 for batch in first)
    assert sorted(torch.cat([batch[1] for batch in first]).tolist()) == list(range(5))
    assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
               for a, b in zip(first, second))
    