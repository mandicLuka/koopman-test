import tensorflow as tf
from keras import backend
class SequenceSquareLoss:

    def __init__(self, gamma=1, **kwargs):
        self.gamma = gamma

    def __call__(self, y, y_pred):
        dim = tf.shape(y)[1]
        gamma = self._gamma_vec(dim)
        square = tf.math.squared_difference(y, y_pred)
        discounted = tf.einsum("ijk, j->ik", square, gamma)

        return backend.mean(discounted, axis=-1)

    def _gamma_vec(self, dim):
        gamma = tf.Variable(tf.ones(dim))
        for i in range(dim-1):
            gamma[i+1].assign(gamma[i] * self.gamma)
        return gamma
