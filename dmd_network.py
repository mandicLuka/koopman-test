import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from mishmash_layer import MishmashLayer


class CoordinateTransformNetwork(keras.Model):


    def __init__(self, dim):
        super(CoordinateTransformNetwork, self).__init__()
        self.out_dim = dim
        self.d1 = layers.Dense(128, activation='relu')
        self.d2 = layers.Dense(64, activation='sigmoid')

        self.embedding = layers.Dense(50, activation="relu")

        self.r1 = layers.Dense(64, activation='sigmoid')
        self.r2 = layers.Dense(128, activation='relu')
        self.restored = layers.Dense(self.out_dim)


        ## koopman definition
        self.U = layers.Dense(50, activation='linear', use_bias=False)

    def get_config(self):
        return { "dim": self.out_dim }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @tf.function
    def embed(self, x):
        e = self.d2(self.d1(x))
        return self.embedding(e)

    @tf.function
    def inverse(self, embed):
        r = self.r2(self.r1(embed))
        return self.restored(r)
        
    def call(self, x):
        e = self.embed(x)
        koopman = self.U(e)
        return self.inverse(koopman)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.

        x = data[0]
        x_p = data[1]


        with tf.GradientTape() as tape:

            # embed = self.embed(x)  # Forward pass
            # next_embed = tf.stop_gradient(self.embed(x_p))

            # koopman = self.U(embed)
            # inverse = self.inverse(koopman)

            # y_pred = (inverse, koopman)
            # y = (x_p, next_embed)

            y_pred = self.call(x)
            y = x_p

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


class MishmashNetwork(keras.Model):


    def __init__(self, dim):
        super(MishmashNetwork, self).__init__()
        self.out_dim = dim
        self.m1 = MishmashLayer(64)
        self.m2 = MishmashLayer(64)
        self.m3 = MishmashLayer(64)

        ## koopman definition
        self.U = layers.Dense(128, activation='linear')

    def get_config(self):
        return { "dim": self.out_dim }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @tf.function
    def embed(self, x):
        e = self.m3(self.m2(self.m1(x)))
        return tf.concat(e, axis=1)

    @tf.function
    def inverse(self, embed):
        embed = tf.split(embed, 2, axis=1)
        r = self.m1.inverse(self.m2.inverse(self.m3.inverse(embed)))
        return r
        
    def call(self, x):
        e = self.embed(x)
        koopman = self.U(e)
        return self.inverse(koopman)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.

        x = data[0]
        x_p = data[1]


        with tf.GradientTape() as tape:

            # embed = self.embed(x)  # Forward pass
            # next_embed = tf.stop_gradient(self.embed(x_p))

            # koopman = self.U(embed)
            # inverse = self.inverse(koopman)

            # y_pred = (inverse, koopman)
            # y = (x_p, next_embed)

            y_pred = self.call(x)
            y = x_p

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
