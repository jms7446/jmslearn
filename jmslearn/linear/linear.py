import numpy as np
from numpy import log
from numpy.linalg import inv, pinv, det
from functools import reduce

from ..util import linv, cv_, norm2, norm2s, get_onehot_encoding, sigmoid
from ..preprocess import IndexEncoder, PolynomialFeature

# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import Pipeline
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDASklearn


class LinearRegression:
    def __init__(self):
        self.w = None
        self.std = None

    def fit(self, X, t):
        N = len(t)
        self.w = linv(X) @ t
        error_vec = X @ self.w - t
        self.std = np.sqrt(np.linalg.norm(error_vec) ** 2 / N)

    def predict(self, X, return_std=False):
        y = X @ self.w
        if return_std:
            return y, self.std
        else:
            return y


class RidgeRegression:
    def __init__(self, alpha=1e-3):
        self.alpha = alpha
        self.w = None
        self.std = None

    def fit(self, X, t):
        N, M = X.shape
        self.w = np.linalg.inv(X.transpose() @ X + self.alpha * np.eye(M)) @ X.transpose() @ t
        self.std = np.mean((X @ self.w - t) ** 2)

    def predict(self, X, return_std=False):
        y = X @ self.w
        if return_std:
            return y, self.std
        else:
            return y


class BayesianRegression:
    def __init__(self, alpha=1., beta=100.):
        self.alpha = alpha
        self.beta = beta
        self.mean = None
        self.precision = None
        self.cov = None
        self.M = None

    def init_prior(self, m):
        self.mean = cv_(np.zeros(m))
        self.precision = self.alpha * np.eye(m)
        self.M = m

    def fit(self, X, t):
        if len(t) == 0:
            return
        t = cv_(t)
        if self.M is None:
            self.init_prior(X.shape[1])
        precision_N = self.precision + self.beta * X.T @ X
        cov_N = inv(precision_N)
        self.mean = cov_N @ (self.precision @ self.mean + self.beta * X.T @ t)
        self.precision = precision_N
        self.cov = cov_N

    def w_mean(self):
        return self.mean.squeeze()

    def w_cov(self):
        return self.cov

    def predict(self, X, sample_size=None, return_std=False):
        if sample_size:
            w_samples = [self.cov @ cv_(np.random.randn(self.M)) + self.mean for _ in range(sample_size)]
            return np.concatenate([X @ w for w in w_samples], axis=1)
        t = (X @ self.mean).squeeze()
        if return_std:
            std = np.sqrt(1 / self.beta + np.sum(X @ self.cov * X, axis=1))
            return t, std
        return t


class EmpiricalBayesRegression(BayesianRegression):
    def __init__(self, alpha=1., beta=1., max_iter=100, verbose=False):
        super().__init__(alpha, beta)
        self.trn_log_evidence_ = None
        self.M = None
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, X, y):
        N, M = X.shape
        y = cv_(y)
        XtX = X.T @ X
        evs = np.linalg.eigvalsh(XtX)
        alpha = self.alpha
        beta = self.beta
        m_N = None
        A = None
        for i in range(self.max_iter):
            pre_params = [alpha, beta]

            A = alpha * np.eye(M) + beta * XtX
            m_N = beta * pinv(A) @ X.T @ y

            lam = beta * evs
            gamma = np.sum(lam / (lam + alpha))
            alpha = gamma / norm2s(m_N)
            beta = (N - gamma) / norm2s(X @ m_N - y)
            if np.allclose(pre_params, [alpha, beta], rtol=1e-8, atol=1e-8):
                break
        if self.verbose:
            print(f"{M - 1}: alpha: {alpha:.5f}, beta: {beta:.5f}")

        self.M = M
        self.alpha = alpha
        self.beta = beta
        self.mean = m_N
        self.precision = A
        self.cov = pinv(A)
        self.trn_log_evidence_ = self._log_evidence(X, y)

    def score(self, X=None, y=None):
        if X is None:
            return self.trn_log_evidence_
        else:
            return self._log_evidence(X, y)

    def _log_evidence(self, X, y):
        N, M = X.shape
        y = cv_(y)
        m_N = self.mean
        A = self.precision
        E_m_N = self.beta / 2 * norm2s(y - X @ m_N) + self.alpha / 2 * norm2s(m_N)
        log_evidence = (0
                        + M / 2 * log(self.alpha)
                        + N / 2 * log(self.beta)
                        - E_m_N
                        - 1 / 2 * log(det(A))
                        - N / 2 * log(2 * np.pi)
                        )
        return log_evidence


class LeastSquaresClassifier:
    def __init__(self):
        self.W = None

    def fit(self, X, y):
        T = get_onehot_encoding(y)
        self.W = pinv(X) @ T

    def predict(self, X):
        Y = X @ self.W
        return np.argmax(Y, axis=1)


class FishersLinearDiscriminant:
    def __init__(self):
        self.w = None

    def fit(self, X, y):
        Xs = [X[y == 0], X[y == 1]]
        ms = [np.mean(Xk, axis=0) for Xk in Xs]
        Ss = [(Xk - m).T @ (Xk - m) for Xk, m in zip(Xs, ms)]
        S_w = reduce(lambda a, b: a + b, Ss)

        self.w = pinv(S_w) @ (ms[1] - ms[0])
        self.w = self.w / np.linalg.norm(self.w, 2)

    def predict(self, X):
        return (X @ self.w > 0).astype(int)


class LinearDiscriminantAnalysis:
    def __init__(self):
        self.encoder = IndexEncoder()
        self.means = None
        self.cov = None
        self.precision = None
        self.W = None
        self.W0 = None
        self.K = None
        self.Ps = None
        self.covs = None

    def fit(self, X, y):
        N, _ = X.shape
        t = self.encoder.fit_transform(y)
        K = np.max(t) + 1

        Xs = [X[t == i, :] for i in range(K)]
        Ns = [Xk.shape[0] for Xk in Xs]
        Ps = [Nk / N for Nk in Ns]
        means = [np.mean(Xk, axis=0, keepdims=True).T for Xk in Xs]
        covs = [(Xk - mean.T).T @ (Xk - mean.T) / Nk for Xk, mean, Nk in zip(Xs, means, Ns)]
        cov = sum(cov * Pk for cov, Pk in zip(covs, Ps))
        precision = pinv(cov)
        W = np.concatenate([precision @ mean for mean in means], axis=1)
        W0 = np.concatenate([-0.5 * mean.T @ precision @ mean + log(pk) for mean, pk in zip(means, Ps)], axis=1)

        self.means = means
        self.cov = cov
        self.precision = precision
        self.W = W
        self.W0 = W0
        self.K = K
        self.Ps = Ps
        self.covs = covs

    def predict(self, X):
        t = np.argmax(X @ self.W + self.W0, axis=1)
        return self.encoder.reverse_transform(t)

    def generate(self, sample_size=10):
        ks = np.random.choice(self.K, size=sample_size, p=self.Ps)
        return np.array([np.random.multivariate_normal(self.means[k].squeeze(), self.covs[k]) for k in ks]), ks


class LogisticRegression:
    def __init__(self, solver="gd", alpha=.1, eta=1e-4, max_iter=1000, rtol=1e-5, atol=1e-5, fit_intercept=True):
        self.solver = solver
        self.encoder = IndexEncoder()
        self.alpha = alpha
        self.eta = eta
        self.max_iter = max_iter
        self.rtol = rtol
        self.atol = atol
        self.fit_intercept = fit_intercept
        self.w = None
        self.b = None

    def fit(self, X, y):
        solver_map = {
            "gd": self._gd_solver,
            "nr": self._nr_solver,
        }
        if self.fit_intercept:
            X = PolynomialFeature(1).transform(X)
        y = self.encoder.fit_transform(y)
        solver_map[self.solver](X, y)

    def _get_decay_mask(self, lw):
        decay_mask = np.ones(lw)
        if self.fit_intercept:
            decay_mask[0] = 0
        return decay_mask

    def _gd_solver(self, X, t):
        N, M = X.shape
        t = cv_(t)
        w = cv_(np.random.normal(0, 1, M))
        pre_y = np.zeros(N)
        converge_thres = norm2(t) * self.rtol
        change = 0
        decay_mask = self._get_decay_mask(len(w))
        for i in range(self.max_iter):
            gradient = X.T @ (sigmoid(X @ w) - t)
            w += - self.eta * gradient + cv_(self.alpha * decay_mask) * w
            y = sigmoid(X @ w)
            change = norm2(y - pre_y)
            if change <= converge_thres:
                print(f"{i} step converged. break, {change}, {converge_thres}")
                break
            pre_y = y
            loss = np.mean(np.square(sigmoid(X @ w) - t))
        else:
            print(f"does not converge, {change}, {converge_thres}")
        self.w = w

    def _nr_solver(self, X, t):
        """FIXME 가끔 H가 singular maxtrix가 되면서 잘못된 값으로 수렴한다. 원인 확인 필요"""
        N, M = X.shape
        t = cv_(t)
        w = cv_(np.random.normal(0, 1, M))
        pre_loss = 0
        decay_mask = self._get_decay_mask(M)
        for i in range(self.max_iter):
            y = sigmoid(X @ w)
            g = X.T @ (y - t)
            r = np.abs(y * np.abs(1 - y)).squeeze()
            H = (X.T * r) @ X + np.diag(self.alpha * decay_mask)
            w -= pinv(H) @ g
            loss = np.mean(np.square(sigmoid(X @ w) - t))
            if abs(loss - pre_loss) <= self.atol:
                print(f"{i} step converged. break, {loss}")
                break
            pre_loss = loss
        self.w = w

    def predict(self, X):
        y = (self.proba(X) >= 0.5).astype(int)
        return self.encoder.reverse_transform(y)

    def proba(self, X):
        if self.fit_intercept:
            X = PolynomialFeature(1).transform(X)
        p = sigmoid(X @ self.w).squeeze()
        return p

    def score(self, X, y):
        N, _ = X.shape
        prd = self.predict(X)
        return np.sum(prd == y) / N


class BayesianLogisticRegression(LogisticRegression):
    def __init__(self, solver="gd", alpha=1., eta=0.1, max_iter=100, fit_intercept=True):
        super().__init__(solver, alpha=alpha, eta=eta, max_iter=max_iter, fit_intercept=fit_intercept)
        self.Sn = None

    def hessian(self, X, y):
        decay_mask = self._get_decay_mask(len(self.w))
        y = sigmoid(X @ self.w)
        r = np.abs(y * (1 - y)).squeeze()
        H = (X.T * r) @ X + np.diag(self.alpha * decay_mask)
        return H

    def fit(self, X, y):
        if self.fit_intercept:
            X = PolynomialFeature(1).transform(X)
        y = self.encoder.fit_transform(y)
        self._gd_solver(X, y)
        self.Sn = pinv(self.hessian(X, y))

    def proba(self, X):
        if self.fit_intercept:
            X = PolynomialFeature(1).transform(X)
        mu = X @ self.w
        var = np.sum((X @ self.Sn) * X, axis=1, keepdims=True)
        kappa = 1 / np.sqrt(1 + np.pi * var / 8)
        return sigmoid(kappa * mu).squeeze()

    def predict(self, X):
        y = (self.proba(X) >= 0.5).astype(int)
        return self.encoder.reverse_transform(y)
