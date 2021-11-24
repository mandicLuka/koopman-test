import tensorflow as tf
from keras import backend

class SequenceSquareLoss:

    def __init__(self, gamma=1, lam=100, **kwargs):
        self.gamma = gamma
        self.lam = lam

    def __call__(self, y, y_pred):
        dim = tf.shape(y)[1]
        result = tf.math.squared_difference(y, y_pred)
        if dim > 1:
            gamma = self._gamma_vec(dim)
            result = tf.einsum("ijk, j->ik", result, gamma)

        return backend.mean(result, axis=-1)

    def _gamma_vec(self, dim):
        gamma = tf.Variable(tf.ones(dim))
        for i in range(dim - 1):
            gamma[i+1].assign(gamma[i] * self.gamma)
        return gamma
