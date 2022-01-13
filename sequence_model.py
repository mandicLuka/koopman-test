from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class SequenceModelNetwork(tf.keras.Model):

    def __init__(self, input_shape, predict_horizon=1, **kwargs):
        super().__init__()
        self.input_window = input_shape[0]
        self.feature_shape = input_shape[1:]
        self.num_features = np.prod(self.feature_shape)
        self.predict_horizon = predict_horizon

        # predictions will be passed as input because
        # the loss function can be complex 
        # self.inputs = [layers.Input(input_shape)]
    
    @abstractmethod
    def embed(self, x):
        pass

    @abstractmethod
    def propagate(self, embedded):
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

        with tf.GradientTape() as tape:

            encoded = self.inverse(self.embed(data))
            preds = []
            state = self.embed(data)
            for _ in range(self.predict_horizon):
                next_state = self.propagate(state)
                inv = self.inverse(next_state)
                preds.append(inv)
                state = next_state

            predictions = tf.stack(preds, axis=1)
            loss = self.compiled_loss((labels, data[:, -1]), (predictions, encoded), regularization_losses=self.losses)

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
        data, labels = data

        encoded = self.inverse(self.embed(data))
        preds = []

        for _ in range(self.predict_horizon):
            next_state = self.propagate(state)
            inv = self.inverse(next_state)
            preds.append(inv)
            state = next_state

        predictions = tf.stack(preds, axis=1)

        # Updates the metrics tracking the loss
        self.compiled_loss((labels, data[:, -1]), (predictions, encoded), regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state((labels, data[:, -1]), (predictions, encoded))
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
