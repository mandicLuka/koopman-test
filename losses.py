import tensorflow as tf
from keras import backend
from sequence_model import SequenceModelNetwork

class DummyZeroLoss(tf.keras.losses.Loss):

    def __init__(self, **kwargs):
        super().__init__(reduction=tf.keras.losses.Reduction.AUTO, name="dummy_zero")

    def call(self, y_true, y_pred):
        return 0

class SequenceMse(tf.keras.losses.Loss):

    def __init__(self, loss_mask=None, **kwargs):
        self.mask = loss_mask
        super().__init__(reduction=tf.keras.losses.Reduction.AUTO, name="sequence_mse")

    def call(self, y_true, y_pred):

        labels = y_true
        predictions = y_pred
        end_dim = len(tf.shape(labels)) - 1
        if self.mask is not None:
            labels = tf.boolean_mask(labels, self.mask, axis=end_dim)
            predictions = tf.boolean_mask(predictions, self.mask, axis=end_dim)

        sq_diff = tf.math.squared_difference(labels, predictions)

        return tf.math.reduce_mean(sq_diff)