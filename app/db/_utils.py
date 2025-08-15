import numpy as np

def normalize(scores):
    if not scores:
        return []

    scores = np.array(scores, dtype=np.float32)
    mean = scores.mean()
    std = scores.std()

    lower = mean - 3 * std
    upper = mean + 3 * std

    if upper == lower:
        return np.ones_like(scores) * 0.5

    normalized = (scores - lower) / (upper - lower)
    normalized = np.clip(normalized, 0, 1)
    return normalized.tolist()