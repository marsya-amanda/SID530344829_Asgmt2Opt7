import numpy as np
from foley.similarity import aligned_rmse, aligned_rmse_fft

def test_fft_matches_roll():
    rng=np.random.default_rng(1); a=rng.normal(size=(3,17)); b=rng.normal(size=(3,17))
    vals=[(np.sqrt(np.mean((a-np.roll(b,s,axis=-1))**2)),s) for s in range(-3,4)]
    assert np.isclose(aligned_rmse_fft(a,b,3)[0],min(vals)[0],atol=1e-6)

def test_identity_and_roll():
    a=np.random.default_rng(2).normal(size=(2,20))
    assert aligned_rmse(a,a,5)<1e-6
    assert aligned_rmse(a,np.roll(a,4,axis=-1),5)<1e-6