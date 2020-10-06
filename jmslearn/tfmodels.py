import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from prmlmy.util import sigmoid, cv_

tf.keras.backend.set_floatx('float64')


class TFLogisticRegression(Model):
    def __init__(self, M, fit_intercept=True, seed=1234):
        super().__init__()
        self.W = tf.Variable(np.random.randn(M).reshape(M, 1))
        if fit_intercept:
            self.b = tf.Variable(np.asarray(1, dtype=np.float64))
        else:
            self.b = tf.Variable(np.asarray(-0.46898513, dtype=np.float64), trainable=False)

    def call(self, x):
        # return tf.sigmoid(tf.add(tf.matmul(x, self.W), self.b))
        return tf.sigmoid(x @ self.W + self.b)


def mean_square_error(prd_y, y):
    return tf.reduce_mean(tf.square(prd_y - y))


def cross_entropy(prd_y, y):
    return -tf.reduce_sum(y * tf.math.log(prd_y) + (1 - y) * tf.math.log(1 - prd_y))


def optimize(var_and_grads, eta):
    for var, grad in var_and_grads:
        var.assign_sub(eta * grad)


class LogisticRegression:
    def __init__(self, fit_intercept=True, max_iter=100, eta=0.01):
        super().__init__()
        self.tf_model = None
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.eta = eta
        self.history = None
        self.W = np.asarray([0.1, 0.1]).reshape(-1, 1)
        self.b = np.asarray([0.0])

    def fit(self, X, y):
        # X = X[:2]
        # y = y[:2]
        t = y.reshape(-1, 1)
        N, M = X.shape
        model = TFLogisticRegression(M, fit_intercept=self.fit_intercept)
        self.history = []
        pre_loss = 0
        for i in range(self.max_iter):
            with tf.GradientTape() as tape:
                loss = cross_entropy(model(X), y)
            # grads = tape.gradient(loss, model.trainable_variables)
            tvs = [model.W, model.b]
            grads = tape.gradient(loss, tvs)
            # print(f"w: {model.W.numpy()}")
            # print(f"b: {model.b.numpy()}")
            # print(f"x: {X}")
            # print(f"y: {y}")
            # print(f"p: {model(X)}")
            # print(f"g: {grads[0]}")
            # print(f"g: {grads[1]}")
            # aslkdjf
            # optimize(zip(model.trainable_variables, grads), self.eta)
            d = (model(X) - t)

            # d = (sigmoid(X @ self.W + self.b) - t)
            # gw = X.T @ d
            # gb = np.sum(d)
            # self.W -= self.eta * gw
            # self.b -= self.eta * gb

            # print(model(X).shape, d.shape, gw.shape)
            # gw = X.T @ d
            # gb = np.sum(d)
            gw = grads[0]
            gb = grads[1]
            # print(gw.shape, gb.shape, model.W.shape, model.b.shape)
            model.W.assign_sub(self.eta * gw)
            model.b.assign_sub(self.eta * gb)
            ws = model.W.numpy().reshape(1, -1)
            bs = model.b.numpy().reshape(1, 1)
            # print(f"w: {model.W.numpy()}")
            # print(loss)
            # ws = self.W.reshape(1, -1)
            # bs = self.b.reshape(1, 1)
            self.history.append(np.concatenate([ws, bs], axis=1))
            if np.isclose(pre_loss, loss, rtol=1e-8, atol=1e-9) and False:
                print(f"{i} step, loss: {loss}, break")
                break
            pre_loss = loss
        self.tf_model = model

    def predict(self, x):
        x = np.asarray(x)
        p = self.tf_model(x).numpy()
        # p = sigmoid(x @ self.W + self.b)
        y = (p > 0.5).astype(int)
        return y


if __name__ == "__main__":
    x = np.asarray([[1.0, 2], [2, 3]])
    print(TFLogisticRegression(2)(x))



