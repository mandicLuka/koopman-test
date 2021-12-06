from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class SequenceModelNetwork(tf.keras.Model):

    def __init__(self, input_shape, **kwargs):
        super().__init__()
        self.input_window = input_shape[0]
        self.feature_shape = input_shape[1:]
        self.num_features = np.prod(self.feature_shape)

        # predictions will be passed as input because
        # the loss function can be complex 
        # self.inputs = [layers.Input(input_shape)]
    
    @abstractmethod
    def embed(self, x):
        pass

    @abstractmethod
    def inverse(self, embed):
        pass
        
    @abstractmethod
    def call(self, x):
        pass

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def train_step(self, data):

        data, labels = data

        predictions = tf.Variable(tf.zeros_like(labels))
        data_var = tf.Variable(data)

        num_labels = tf.shape(labels)[1]
        with tf.GradientTape() as tape:

            # embed = self.embed(x)  # Forward pass
            # next_embed = tf.stop_gradient(self.embed(x_p))

            # koopman = self.U(embed)
            # inverse = self.inverse(koopman)

            # y_pred = (inverse, koopman)
            # y = (x_p, next_embed)
            encoded = self.inverse(self.embed(data))
            if num_labels == 1:
                predictions = tf.expand_dims(self.call(data)[:, -1], axis=1)
            else:
                preds = []
                for i in range(num_labels):
                    pred = self.call(data)
                    data_var[:, 0:-1].assign(data[:, 1:]) 
                    data_var[:, -1, :].assign(pred[:, -1], 1)
                    preds.append(pred[:, -1])

                predictions = tf.stack(preds, axis=1)

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss((labels, data), (predictions, encoded), regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state((labels, data), (predictions, encoded))
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data
        data, labels = data

        predictions = tf.Variable(tf.zeros_like(labels))
        data_var = tf.Variable(data)

        num_labels = tf.shape(labels)[1]
        encoded = self.inverse(self.embed(data))
        if num_labels == 1:
            predictions = tf.expand_dims(self.call(data)[:, -1], axis=1)
        else:
            preds = []
            for i in range(num_labels):
                pred = self.call(data)
                data_var[:, 0:-1].assign(data[:, 1:]) 
                data_var[:, -1, :].assign(pred[:, -1], 1)
                preds.append(pred[:, -1])

            predictions = tf.stack(preds, axis=1)

        # Updates the metrics tracking the loss
        self.compiled_loss((labels, data), (predictions, encoded), regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state((labels, data), (predictions, encoded))
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
