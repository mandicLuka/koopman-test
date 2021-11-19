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
        with tf.GradientTape() as tape:

            # embed = self.embed(x)  # Forward pass
            # next_embed = tf.stop_gradient(self.embed(x_p))

            # koopman = self.U(embed)
            # inverse = self.inverse(koopman)

            # y_pred = (inverse, koopman)
            # y = (x_p, next_embed)
            preds = []
            for i in range(tf.shape(labels)[1]):
                pred = self.call(data)
                data_var[:, 0:-1].assign(data[:, 1:]) 
                data_var[:, -1, :].assign(pred[:, -1], 1)
                preds.append(pred[:, -1])

            predictions = tf.stack(preds, axis=1)
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(labels, predictions, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(labels, predictions)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


class MishmashNetwork(keras.Model):


    def __init__(self, dim, **kwargs):
        super(MishmashNetwork, self).__init__()
        self.out_dim = dim
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
        e = self.m3(self.m2(self.m1([x, x])))
        return tf.concat(e, axis=1)

    @tf.function
    def inverse(self, embed):
        embed = tf.split(embed, 2, axis=1)
        r = self.m1.inverse(self.m2.inverse(self.m3.inverse(embed)))
        return (r[0] + r[1]) / 2
        
    def call(self, x):
        e = self.embed(x)
        koopman = self.U(e)
        return self.inverse(koopman)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        self.step += 1
        with tf.profiler.experimental.Trace('train', step_num=self.step, _r=1):

            x = data[0]
            x_p = data[1]


            with tf.GradientTape() as tape:

                # embed = self.embed(x)  # Forward pass
                # next_embed = tf.stop_gradient(self.embed(x_p))

                # koopman = self.U(embed)
                # inverse = self.inverse(koopman)

                # y_pred = (inverse, koopman)
                # y = (x_p, next_embed)

                # y_pred = self.call(x)
                # y = x_p

                x_embed = self.embed(x)
                x_p_embed = self.embed(x_p)
                y = x_p_embed
                y_pred = self.U(x_embed)

                # Compute the loss value
                # (the loss function is configured in `compile()`)
                loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
                a = 5

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            # Update metrics (includes the metric that tracks the loss)
            self.compiled_metrics.update_state(y, y_pred)
            # Return a dict mapping metric names to current value
            return {m.name: m.result() for m in self.metrics}
