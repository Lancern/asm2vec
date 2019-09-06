import numpy as np


def make_small_ndarray(dim: int) -> np.ndarray:
    rng = np.random.default_rng()
    return (rng.random(dim) - 0.5) / dim
