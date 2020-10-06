import numpy as np
from numpy import asarray
from numpy.linalg import norm


def linv(X):
    return np.linalg.pinv(X.transpose() @ X) @ X.transpose()


def rinv(X):
    return X.transpose() @ np.linalg.pinv(X @ X.transpose())


def cv_(x):
    x = asarray(x)
    if x.ndim == 1:
        return x[:, None]
    elif x.ndim == 2:
        if x.shape[1] == 1:
            return x
    raise ValueError(f"Wrong input: {x}, shape: {x.shape}")


def norm2s(x):
    return norm(x, 2) ** 2


def norm2(x):
    return norm(x, 2)


def calc_range(x, pad_ratio=0.0):
    x_min = np.min(x)
    x_max = np.max(x)
    pad = pad_ratio * (x_max - x_min)
    return x_min - pad, x_max + pad


def get_onehot_encoding(x):
    M = max(x) + 1
    if M > 100:
        raise ValueError(f"dim > 1000 is not allowed ({M})")
    x = asarray(x).reshape(-1)
    return np.eye(M)[x]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
