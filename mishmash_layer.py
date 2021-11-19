import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer

class MishmashLayer(Layer):

    def __init__(self, num_outputs, activation='relu'):
        super(MishmashLayer, self).__init__()
        self.num_outputs = num_outputs
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        u1_shape, u2_shape = input_shape
        
        self.u1_shape = u1_shape
        self.u2_shape = u2_shape
        self.bias = self.add_weight("upper_b", shape=[u2_shape[-1], self.num_outputs], initializer=tf.keras.initializers.RandomNormal(stddev=0.005))
        # self.coef = self.add_weight("upper_c", shape=[u2_shape[-1], u1_shape[-1], self.num_outputs], initializer=tf.keras.initializers.RandomNormal(stddev=0.0005))
        self.matrix_bias = tf.constant(np.ones(shape=[u1_shape[-1], self.num_outputs]), dtype="float32")
        self.coef1 = self.add_weight("upper_c1", shape=[u2_shape[-1], self.num_outputs*u1_shape[-1]], initializer=tf.keras.initializers.RandomNormal(stddev=0.0005))

        # u2 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype="float32")
        # u1 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype="float32")
        # fw = self.fwd(u1, u2)
        # bwd = self.bwd(u2, fw)

        # a = tf.matmul(u2, self.bias)
        # b = self.activation(tf.tensordot(u2, self.coef, axes=1)) + self.matrix_bias
        # aa = tf.expand_dims(u2, 1)
        # c = tf.map_fn(lambda x: tf.matmul(x[0], x[1])[0], (aa, b), fn_output_signature="float32")
        # b = tf.matmul(u2, (self.activation(tf.tensordot(u2, self.coef, axes=1))) + self.matrix_bias)


        # v2 = fw
        # v1 = u2
        # b = v2 - self.activation(tf.matmul(v1, self.bias))
        # inv_activ = tf.map_fn(lambda x: tf.linalg.pinv(x), \
        #     self.activation(tf.tensordot(v1, self.coef, axes=1)) + self.matrix_bias)
        # bb = tf.expand_dims(v1, 1)

        # # elementwise do matrix product between v1 and inv_activ
        # final = tf.map_fn(lambda x: tf.matmul(x[0], x[1])[0], (b, inv_activ), fn_output_signature="float32")

    @tf.function
    def fwd(self, u1, u2):
        # b = self.activation(tf.matmul(u2, self.bias))
        # activ = self.activation(tf.tensordot(u2, self.coef, axes=1)) + self.matrix_bias
        # u1_expand = tf.expand_dims(u1, 1)

        # # elementwise do matrix product between u1 and activ
        # inner = tf.map_fn(lambda x: tf.matmul(x[0], x[1])[0], (u1_expand, activ), fn_output_signature="float32")

        # return b + inner


        b = self.activation(tf.matmul(u2, self.bias))
        n = self.activation(tf.matmul(u2, self.coef1))
        activ = tf.reshape(n, (-1, self.u1_shape[-1], self.num_outputs)) + self.matrix_bias

        ret = tf.map_fn(lambda x: tf.matmul(x[0], x[1]), (tf.expand_dims(u1, 1), activ), fn_output_signature="float32")
        ret = b + ret[:, 0]
        return ret

    @tf.function
    def bwd(self, v1, v2):
        # b = v2 - self.activation(tf.matmul(v1, self.bias))
        # inv_activ = tf.map_fn(lambda x: tf.linalg.pinv(x), \
        #     self.activation(tf.tensordot(v1, self.coef, axes=1)) + self.matrix_bias)
        # b_expand = tf.expand_dims(b, 1)

        # # elementwise do matrix product between v1 and inv_activ
        # final = tf.map_fn(lambda x: tf.matmul(x[0], x[1])[0], (b_expand, inv_activ), fn_output_signature="float32")

        b = v2 - self.activation(tf.matmul(v1, self.bias))
        n = self.activation(tf.matmul(v1, self.coef1))
        activ = tf.reshape(n, (-1, self.u1_shape[-1], self.num_outputs)) + self.matrix_bias

        inv = tf.map_fn(lambda x: tf.linalg.pinv(x), activ)
        ret = tf.map_fn(lambda x: tf.matmul(x[0], x[1]), (tf.expand_dims(b, 1), inv), fn_output_signature="float32")
        return ret[:, 0]

    def call(self, inputs):
        u1, u2 = inputs
        v1 = u2
        v2 = self.fwd(u1, u2)
        return [v1, v2]

    def inverse(self, inputs):
        v1, v2 = inputs
        u1 = self.bwd(v1, v2)
        u2 = v1
        return [u1, u2]
