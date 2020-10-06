import numpy as np
from numpy import asarray
import itertools
import functools

from .base import TransformerMixin


class PolynomialFeature:
    def __init__(self, degree=2):
        self.degree = degree

    def transform(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        x_t = x.transpose()
        features = [np.ones(len(x))]
        for degree in range(1, self.degree + 1):
            for items in itertools.combinations_with_replacement(x_t, degree):
                features.append(functools.reduce(lambda a, b: a * b, items))
        return np.asarray(features).transpose()


class GaussianFeature:
    def __init__(self, means, var):
        self.means = np.asarray(means)
        self.var = var

    def transform(self, x):
        x = np.asarray(x)
        features = [np.ones(len(x))]
        features.extend([self._normal(x, mean, self.var) for mean in self.means])
        return asarray(features).transpose()

    @staticmethod
    def _normal(x, mean, var):
        return np.exp(-(x - mean) ** 2 / (2 * var))


class SigmoidalFeature:
    def __init__(self, means, coef):
        self.means = asarray(means)
        self.coef = coef

    def transform(self, x):
        x = asarray(x)
        features = [np.ones(len(x))]
        features.extend([self._sigmoid((x - mean) * self.coef) for mean in self.means])
        return asarray(features).transpose()

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))


class IndexEncoder(TransformerMixin):
    def __init__(self, values=None):
        self.decode_map = None
        self.encode_map = None
        if values:
            self.fit(values)

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.decode_map = {idx: val for idx, val in enumerate(np.sort(np.unique(X.flatten())))}
        self.encode_map = {val: idx for idx, val in self.decode_map.items()}

    def transform(self, X):
        return np.vectorize(self.encode_map.get)(X)

    def reverse_transform(self, X):
        return np.vectorize(self.decode_map.get)(X)




