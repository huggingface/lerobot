import numpy as np


def symlog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(abs(x))


def symexp(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * (np.exp(np.abs(x)) - 1)
