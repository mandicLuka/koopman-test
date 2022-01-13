import tensorflow as tf
from tensorflow import keras
import numpy as np
from mishmash_layer import MishmashLayer
from sequence_model import SequenceModelNetwork



class CoordinateTransformNetwork(SequenceModelNetwork):


    def __init__(self, input_shape, layers=None, **kwargs):
        super().__init__(input_shape, **kwargs)
        
        self.in_layers = layers
        layers = layers or [32, 32]

        self.fwd_layers = []
        for l in layers[:-1]:
            self.fwd_layers.append(keras.layers.Dense(l, activation='relu'))
            
        self.embedding = keras.layers.Dense(layers[-1], activation="relu")

        self.bwd_layers = []
        for l in reversed(layers[:-1]):
            self.bwd_layers.append(keras.layers.Dense(l, activation='relu'))

        self.restored = keras.layers.Dense(self.num_features)

        ## koopman definition
        self.U = keras.layers.Dense(layers[-1], activation='linear')

    def get_config(self):
        return { 
            "input_shape": (self.input_window, *self.feature_shape),
            "layers": self.in_layers 
            }

    @tf.function
    def embed(self, x):
        e = tf.reshape(x, (-1, self.input_window * self.num_features))
        for l in self.fwd_layers:
            e = l(e)
        return self.embedding(e)

    @tf.function
    def propagate(self, embedded):
        return self.U(embedded)

    @tf.function
    def inverse(self, embed):
        r = embed
        for l in self.bwd_layers:
            r = l(r)
        r = self.restored(r)
        return r
        # ret = tf.reshape(r, (-1, self.input_window, *self.feature_shape))
        # return tf.reshape(r, (-1, self.input_window, *self.feature_shape))
        
    @tf.function
    def call(self, x):
        e = self.embed(x)
        koopman = self.U(e)
        inverse = self.inverse(koopman)
        return inverse

class MishmashNetwork(SequenceModelNetwork):


    def __init__(self, input_shape, layers=None, **kwargs):
        super().__init__(input_shape)

        self.in_layers = layers
        layers = layers or [32, 32]

        self.mm_layers = []
        for l in layers[:-1]:
            self.mm_layers.append(MishmashLayer(l))

        second_to_last_shape = layers[-2] if len(layers) > 1 \
            else self.num_features * self.input_window

        ## koopman definition
        self.U = keras.layers.Dense(layers[-1] + second_to_last_shape, activation='linear')

    def get_config(self):
        return { 
            "input_shape": (self.input_window, *self.feature_shape),
            "layers": self.in_layers 
            }

    # @tf.function
    def embed(self, x):
        inp = tf.reshape(x, (-1, self.num_features * self.input_window))
        e = [inp, inp]
        for l in self.mm_layers:
            e = l(e)
        return tf.concat(e, axis=1)

    # @tf.function
    def inverse(self, embed):
        embed = tf.split(embed, 2, axis=1)
        r = embed
        for l in reversed(self.mm_layers):
            r = l(r, inverse=True)
        mean = (r[0] + r[1]) / 2
        return tf.reshape(mean, (-1, self.input_window, *self.feature_shape))
        
    def call(self, x):
        e = self.embed(x)
        koopman = self.U(e)
        inverse = self.inverse(koopman)
        return tf.expand_dims(inverse[:, -1, :], axis=1)