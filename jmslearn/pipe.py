

class SimplePipe:
    """Estimator must be placed at the end"""

    def __init__(self, transformers):
        self.transformers = transformers
        self.preprocesses = transformers[:-1]
        self.estimator = transformers[-1]

    def fit(self, X, y=None):
        X = self._transform_pre(X)
        self.estimator.fit(X, y)

    def predict(self, X, *args, **kwargs):
        X = self._transform_pre(X)
        return self.estimator.predict(X, *args, **kwargs)

    def transform(self, X):
        X = self._transform_pre(X)
        return self.estimator.transform(X)

    def _transform_pre(self, X):
        for t in self.preprocesses:
            X = t.transform(X)
        return X

    def get_params_(self):
        return self.estimator.params_

    def score(self, X=None, y=None):
        return self.estimator.score(X, y)
