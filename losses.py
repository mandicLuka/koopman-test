import tensorflow as tf
from keras import backend

class SequenceSquareLoss:

    def __init__(self, gamma=1, lam=100, **kwargs):
        self.gamma = gamma
        self.lam = lam

    def __call__(self, y, y_pred):
        y, data = y
        y_pred, encoded = y_pred

        dim = tf.shape(y)[1]
        loss_koopman = tf.math.squared_difference(y, y_pred)
        if dim > 1:
            gamma = self._gamma_vec(dim)
            loss_koopman = tf.einsum("ijk, j->ik", loss_koopman, gamma)

        loss_autoencoder = tf.math.squared_difference(data, encoded)
        result = loss_koopman + self.lam * loss_autoencoder
        return backend.mean(result, axis=-1)

    def _gamma_vec(self, dim):
        gamma = tf.Variable(tf.ones(dim))
        for i in range(dim - 1):
            gamma[i+1].assign(gamma[i] * self.gamma)
        return gamma
