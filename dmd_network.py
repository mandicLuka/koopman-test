import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from mishmash_layer import MishmashLayer


class CoordinateTransformNetwork(keras.Model):


    def __init__(self, input_shape, **kwargs):
        super(CoordinateTransformNetwork, self).__init__()
        self.input_window = input_shape[0]
        self.feature_shape = input_shape[1:]
        self.num_features = np.prod(self.feature_shape)

        self.d1 = layers.Dense(128, activation='relu')
        self.d2 = layers.Dense(64, activation='sigmoid')

        self.embedding = layers.Dense(50, activation="relu")

        self.r1 = layers.Dense(64, activation='sigmoid')
        self.r2 = layers.Dense(128, activation='relu')
        self.restored = layers.Dense(self.num_features)


        ## koopman definition
        self.U = layers.Dense(50, activation='linear', use_bias=False)

    def get_config(self):
        return { "input_shape": (self.input_window, *self.feature_shape) }


    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @tf.function
    def embed(self, x):
        e = self.d2(self.d1(tf.reshape(x, (-1, self.num_features))))
        return self.embedding(e)

    @tf.function
    def inverse(self, embed):
        r = self.r2(self.r1(embed))
        restored = self.restored(r)
        return tf.reshape(restored, (-1, self.input_window, *self.feature_shape))
        
    def call(self, x):
        e = self.embed(x)
        koopman = self.U(e)
        return self.inverse(koopman)

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
            loss_koopman = self.compiled_loss(labels, predictions, regularization_losses=self.losses)
            loss_autoencoder = self.compiled_loss(data, encoded, regularization_losses=self.losses)
            total_loss = 100*loss_koopman + 70*loss_autoencoder

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(labels, predictions)
        # Return a dict mapping metric names to current value
        return {'loss': total_loss}


class MishmashNetwork(keras.Model):


    def __init__(self, input_shape, **kwargs):
        super(MishmashNetwork, self).__init__()
        self.input_window = input_shape[0]
        self.feature_shape = input_shape[1:]
        self.num_features = np.prod(self.feature_shape)

        self.m1 = MishmashLayer(32)
        self.m2 = MishmashLayer(32)
        self.m3 = MishmashLayer(32)
        self.step = 0

        ## koopman definition
        self.U = layers.Dense(64, activation='linear')

    def get_config(self):
        return { "dim": self.out_dim }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @tf.function
    def embed(self, x):
        inp = tf.reshape(x, (-1, self.num_features))
        e = self.m3(self.m2(self.m1([inp, inp])))
        return tf.concat(e, axis=1)

    @tf.function
    def inverse(self, embed):
        embed = tf.split(embed, 2, axis=1)
        r = self.m1.inverse(self.m2.inverse(self.m3.inverse(embed)))
        mean = (r[0] + r[1]) / 2
        return tf.reshape(mean, (-1, self.input_window, *self.feature_shape))
        
    def call(self, x):
        e = self.embed(x)
        koopman = self.U(e)
        return self.inverse(koopman)

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
            loss_koopman = self.compiled_loss(labels, predictions, regularization_losses=self.losses)
            loss_autoencoder = self.compiled_loss(data, encoded, regularization_losses=self.losses)
            total_loss = 1000*loss_koopman + 100*loss_autoencoder

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(labels, predictions)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
