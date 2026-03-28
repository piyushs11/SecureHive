import sys, os, numpy as np, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.privacy.dp_embeddings import clip_embedding, gaussian_mechanism, privacy_report

def test_clip_reduces_large_norm():
    v = np.ones(384) * 10.0      # norm is very large
    clipped = clip_embedding(v, clip_norm=1.0)
    assert np.linalg.norm(clipped) <= 1.001  # within floating point tolerance

def test_clip_preserves_small_norm():
    v = np.ones(384) * 0.001     # norm already < 1
    original = v.copy()
    clipped = clip_embedding(v, clip_norm=1.0)
    np.testing.assert_array_almost_equal(clipped, original)

def test_gaussian_adds_noise():
    v = np.zeros(384)
    noisy = gaussian_mechanism(v, epsilon=1.0)
    # Should not be all zeros after noise
    assert not np.allclose(noisy, v)

def test_high_epsilon_less_noise():
    """More epsilon = less noise = weaker privacy."""
    np.random.seed(42)
    v = np.ones(384)
    noise_strong = np.std(gaussian_mechanism(v, epsilon=0.1) - v)
    noise_weak   = np.std(gaussian_mechanism(v, epsilon=5.0) - v)
    assert noise_strong > noise_weak

def test_privacy_report_levels():
    assert privacy_report(0.1)["privacy_level"] == "strong"
    assert privacy_report(1.0)["privacy_level"] == "moderate"
    assert privacy_report(9.0)["privacy_level"] == "weak"