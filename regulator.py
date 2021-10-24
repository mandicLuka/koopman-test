from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
import control.control as control

class Regulator(ABC):

    @abstractmethod
    def get_u(self, x_ref, x, dt):
        pass


class PolePlacement(Regulator):

    def __init__(self, A, B, poles):
        K_as_np = control.place(A, B, poles)
        self.K = tf.convert_to_tensor(K_as_np, dtype=tf.float32)

    def get_u(self, x_ref, x, dt):
        return tf.matmul((x_ref - x), self.K)



class PID(Regulator):

    def __init__(self, p, i=0, d=0):
        self.k_p = p
        self.k_i = i
        self.k_d = d
        self.prev_x_diff = None
        self.error_sum = None


    def get_u(self, x_ref, x, dt):
        
        diff = x_ref - x
        p = self.k_p * diff

        error_sum = self.error_sum if isinstance(self.error_sum, tf.Tensor) else 0
        self.error_sum = error_sum + diff * dt
        i = self.k_i * self.error_sum

        prev_x_diff = self.prev_x_diff if isinstance(self.prev_x_diff, tf.Tensor) else 0
        d = self.k_d * (diff - prev_x_diff) / dt  
        self.prev_x_diff = prev_x_diff

        return p + i + d