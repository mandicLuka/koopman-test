import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
import numpy as np
from mishmash_layer import MishmashLayer
from sequence_model import SequenceModelNetwork, ForcedSequenceModelNetwork


class ForcedCoordinateTransformNetwork(ForcedSequenceModelNetwork):


    def __init__(self, input_shape,
            state_layers=None, force_layers=None, **kwargs):
        super().__init__(input_shape, **kwargs)

        self.state_in_layers = state_layers
        state_layers = state_layers or [32, 32, 32]
        force_layers = force_layers or [32, 32]


        ### STATE NETWORK
        self.state_fwd_layers = []
        for l in state_layers[:-1]:
            self.state_fwd_layers.append(keras.layers.Dense(l, activation='relu'))
            
        self.state_embedding = keras.layers.Dense(state_layers[-1], activation="relu")

        self.state_bwd_layers = []
        for l in reversed(state_layers[:-1]):
            self.state_bwd_layers.append(keras.layers.Dense(l, activation='relu'))

        self.state_restored = keras.layers.Dense(self.num_state_features)
        
        ### FORCE NETWORK
        self.force_fwd_layers = []
        for l in force_layers[:-1]:
            self.force_fwd_layers.append(keras.layers.Dense(l, activation='relu'))
            
        self.force_embedding = keras.layers.Dense(force_layers[-1], activation="relu")

        self.force_bwd_layers = []
        for l in reversed(state_layers[:-1]):
            self.force_bwd_layers.append(keras.layers.Dense(l, activation='relu'))

        self.force_restored = keras.layers.Dense(self.num_force_features)

        ## koopman definition
        self.U = keras.layers.Dense(state_layers[-1], activation='linear', use_bias=False, kernel_regularizer=regularizers.l1(self.koopman_l1))

    @property
    def state_embed_dim(self):
        return self.state_in_layers[-1]

    @property
    def force_embed_dim(self):
        return self.force_in_layers[-1]

    def get_config(self):
        return { 
            "input_shape": ((self.input_window, *self.state_shape), self.force_shape),
            "state_layers": self.state_in_layers,
            "force_layers": self.state_in_layers 
            }

    @tf.function
    def embed_state(self, x):
        e = tf.reshape(x, (-1, self.input_window * self.num_state_features))
        for l in self.state_fwd_layers:
            e = l(e)
        return self.state_embedding(e)

    @tf.function
    def embed_force(self, x):
        e = tf.reshape(x, (-1, self.num_force_features))
        for l in self.force_fwd_layers:
            e = l(e)
        return self.force_embedding(e)

    @tf.function
    def propagate(self, data):
        embedded_state, embedded_force = data
        concat = tf.concat((embedded_state, embedded_force), axis=1)
        return self.U(concat)

    @tf.function
    def inverse_state(self, embed):
        r = embed
        for l in self.state_bwd_layers:
            r = l(r)
        r = self.state_restored(r)
        return r

    @tf.function
    def inverse_force(self, embed):
        r = embed
        for l in self.force_bwd_layers:
            r = l(r)
        r = self.force_restored(r)
        return r
        
class CoordinateTransformNetwork(SequenceModelNetwork):


    def __init__(self, input_shape, layers=None, **kwargs):
        super().__init__(input_shape, **kwargs)
        
        self.in_layers = layers
        layers = layers or [32, 32]

        self.fwd_layers = []
        for l in layers[:-1]:
            self.fwd_layers.append(keras.layers.Dense(l, activation='relu', kernel_regularizer='l2'))
            
        self.embedding = keras.layers.Dense(layers[-1], activation="relu", kernel_regularizer='l2')

        self.bwd_layers = []
        for l in reversed(layers[:-1]):
            self.bwd_layers.append(keras.layers.Dense(l, activation='relu', kernel_regularizer='l2'))

        self.restored = keras.layers.Dense(self.num_features, kernel_regularizer='l2')

        ## koopman definition
        self.U = keras.layers.Dense(layers[-1], activation='linear', kernel_regularizer='l2', use_bias=False)

    @property
    def state_embed_dim(self):
        return self.in_layers[-1]


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
        
class MishmashNetwork(SequenceModelNetwork):


    def __init__(self, input_shape, layers=None, **kwargs):
        super().__init__(input_shape, **kwargs)

        self.in_layers = layers
        layers = layers or [32, 32]

        self.splits = []

        self.mm_layers = []
        for i, l in enumerate(layers[:-1]):
            
            split = [self.num_features, l] if i == 0 \
                else [layers[i - 1], l]
            self.mm_layers.append(MishmashLayer(l))
            self.splits.append(split)
        
        self.state_embedding = MishmashLayer(layers[-1])
        self.splits.append([self.splits[-1][1], layers[-1]])

        ## koopman definition
        self.U = keras.layers.Dense(layers[-1] + self.splits[-1][0], activation='linear', use_bias=False)


    @property
    def state_embed_dim(self):
        return self.splits[-1][0] + \
            self.splits[-1][1]

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

    def propagate(self, embedded):
        return self.U(embedded)

    # @tf.function
    def inverse(self, embed):
        embed = tf.split(embed, 2, axis=1)
        r = embed
        for l in reversed(self.mm_layers):
            r = l(r, inverse=True)
        mean = (r[0] + r[1]) / 2
        return tf.reshape(mean, (-1, self.input_window, *self.feature_shape))[:, -1]
        



class ForcedMishmashNetwork(ForcedSequenceModelNetwork):


    def __init__(self, input_shape, state_layers=None, force_layers=None, **kwargs):
        super().__init__(input_shape, **kwargs)

        state_layers = state_layers or [32, 32]
        force_layers = force_layers or [32, 32]
        self.state_in_layers = state_layers
        self.force_in_layers = force_layers

        self.state_splits = []
        self.force_splits = []

        self.state_mm_layers = []
        for i, l in enumerate(state_layers[:-1]):
            
            split = [self.num_state_features, l] if i == 0 \
                else [state_layers[i - 1], l]
            self.state_mm_layers.append(MishmashLayer(l))
            self.state_splits.append(split)
        
        self.state_embedding = MishmashLayer(state_layers[-1])
        self.state_splits.append([self.state_splits[-1][1], state_layers[-1]])

        self.force_mm_layers = []
        for l in force_layers[:-1]:
            split = [self.num_force_features, l] if i == 0 \
                else [force_layers[i - 1], l]
            self.force_mm_layers.append(MishmashLayer(l))
            self.force_splits.append(split)
        
        self.force_embedding = MishmashLayer(force_layers[-1])
        self.force_splits.append([self.force_splits[-1][1], force_layers[-1]])

        ## koopman definition
        self.U = keras.layers.Dense(state_layers[-1] + self.state_splits[-1][0], activation='linear', use_bias=False)


    @property
    def state_embed_dim(self):
        return self.state_splits[-1][0] + \
            self.state_splits[-1][1]

    @property
    def force_embed_dim(self):
        return self.force_splits[-1][0] + \
            self.force_splits[-1][1]

    def get_config(self):
        return { 
            "input_shape": (self.input_window, *self.feature_shape),
            "state_layers": self.state_in_layers,
            "force_layers": self.state_in_layers 
            }

    # @tf.function
    def embed_state(self, x):
        inp = tf.reshape(x, (-1, self.num_state_features * self.input_window))
        e = [inp, inp]
        for l in self.state_mm_layers:
            e = l(e)
        return tf.concat(self.state_embedding(e), axis=1)

    @tf.function
    def embed_force(self, x):
        inp = tf.reshape(x, (-1, self.num_force_features))
        e = [inp, inp]
        for l in self.force_mm_layers:
            e = l(e)
        return tf.concat(self.force_embedding(e), axis=1)

    def propagate(self, embedded):
        state, force = embedded
        concat = tf.concat((state, force), axis=1)
        return self.U(concat)

    @tf.function
    def inverse_state(self, embed):
        embed = tf.split(embed, self.state_splits[-1], axis=1)
        r = self.state_embedding(embed, inverse=True)
        for l in reversed(self.state_mm_layers):
            r = l(r, inverse=True)
        mean = (r[0] + r[1]) / 2
        return tf.reshape(mean, (-1, self.input_window, self.num_state_features))[:, -1]

    @tf.function
    def inverse_force(self, embed):
        embed = tf.split(embed, self.force_splits[-1], axis=1)
        r = self.force_embedding(embed, inverse=True)
        for l in reversed(self.force_mm_layers):
            r = l(r, inverse=True)
        mean = (r[0] + r[1]) / 2
        return tf.reshape(mean, (-1, self.num_force_features))