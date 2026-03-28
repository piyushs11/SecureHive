import numpy as np

def clip_embedding(embedding: np.ndarray, clip_norm: float = 1.0) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    if norm > clip_norm:
        return embedding * (clip_norm / norm)
    return embedding.copy()

def gaussian_mechanism(
    embedding: np.ndarray,
    epsilon: float,
    delta: float = 1e-5,
    clip_norm: float = 1.0,
) -> np.ndarray:
    clipped = clip_embedding(embedding, clip_norm)
    sigma = clip_norm * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon
    noise = np.random.normal(loc=0.0, scale=sigma, size=clipped.shape)
    return clipped + noise

def laplace_mechanism(values: np.ndarray, sensitivity: float, epsilon: float) -> np.ndarray:
    scale = sensitivity / epsilon
    return values + np.random.laplace(loc=0.0, scale=scale, size=values.shape)

def privacy_report(epsilon: float, delta: float = 1e-5, clip_norm: float = 1.0) -> dict:
    sigma = clip_norm * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon
    level = "strong" if epsilon < 1.0 else "moderate" if epsilon < 5.0 else "weak"
    return {
        "epsilon": epsilon,
        "delta": delta,
        "sigma": round(sigma, 4),
        "privacy_level": level,
        "note": f"Each of 384 embedding dims gets N(0, {sigma:.2f}²) noise",
    }