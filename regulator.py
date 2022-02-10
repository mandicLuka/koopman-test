from abc import ABC, abstractmethod, abstractproperty
import tensorflow as tf
import numpy as np
import control.control as control

class Regulator(ABC):

    @abstractproperty
    def dim(self):
        pass

    @abstractmethod
    def get_u(self, x_ref, x, dt):
        pass


    def _divide_A(self, A):
        return A[:self.state_dim].T, A[self.state_dim:].T


class PolePlacement(Regulator):

    def __init__(self, A, poles, B=None, dtype=np.float32):

        A = A.astype(dtype)
        if B is None:
            self.state_dim = A.shape[-1]
            self.input_dim = A.shape[0] - self.state_dim
            self.A, self.B = self._divide_A(A)
        else:
            B = B.astype(dtype)
            self.input_dim = A.shape[0]
            self.input_dim = B.shape[0]
            self.A = A
            self.B = B

        K = control.place(self.A, self.B, poles)
        self.K = K.T.astype(dtype)

    def get_u(self, x_ref, x, dt):
        return np.dot((x - x_ref), self.K), np.dot(x_ref, self.A)

    @property
    def dim(self):
        return int(self.K.shape[1])

class PseudoinverseTracker(Regulator):

    def __init__(self, A, B=None, dtype=np.float32):
        self.dtype = dtype
        A = A.astype(dtype)
        if B is None: 
            self.state_dim = A.shape[-1]
            self.input_dim = A.shape[0] - self.state_dim
            self.A, self.B = self._divide_A(A)
        else:
            B = B.astype(dtype)
            self.input_dim = A.shape[0]
            self.input_dim = B.shape[0]
            self.A = A
            self.B = B

        a = np.linalg.cond(self.A)
        b = np.linalg.cond(self.B)

        ua, sa, vha = np.linalg.svd(self.A)
        ub, sb, vhb = np.linalg.svd(self.B)
        
        self.pinv_B = np.linalg.pinv(self.B.T)

    def update_ref(self, ref_v):
        pass   

    @property
    def dim(self):
        return int(self.B.shape[0])

    def get_u(self, x_ref, x, dt):
        # return np.dot((x_ref - x) * 0.3, self.pinv_B)
        return np.dot((x_ref - np.dot(x, self.A.T)), self.pinv_B)

class TrackingLQR(Regulator):

    def __init__(self, A, B=None, Q=None, R=None, N=None, dtype=np.float32):
        self.dtype = dtype
        A = A.astype(dtype)
        if B is None: 
            self.state_dim = A.shape[-1]
            self.input_dim = A.shape[0] - self.state_dim
            self.A, self.B = self._divide_A(A)
        else:
            B = B.astype(dtype)
            self.input_dim = A.shape[0]
            self.input_dim = B.shape[0]
            self.A = A
            self.B = B

        self.Q = np.zeros((self.state_dim + 1, self.state_dim + 1), dtype=dtype)
        if Q is not None:
            self.Q[:-1, :-1] = Q
        else:
            self.Q[:self.state_dim, :self.state_dim] = np.eye(self.state_dim, dtype=dtype)

        # self.R = R or np.zeros((self.input_dim, self.input_dim), dtype=dtype)
        self.R = R or np.eye(self.input_dim, dtype=dtype) * 1e-4
        self.N = N

    def update_ref(self, ref_v):
        
        sh = self.A.shape
        A_hat = np.zeros((sh[0] + 1, sh[1] + 1))
        A_hat[:sh[0], :sh[1]] = self.A

        sh = self.B.shape
        B_hat = np.zeros((sh[0] + 1, sh[1]))
        B_hat[:sh[0], :sh[1]] = self.B

        self.K = np.zeros((ref_v.shape[0], *self.B.T.shape), self.dtype)
        self.S = np.zeros((ref_v.shape[0], *self.A.shape), self.dtype)
        for i, r in enumerate(ref_v):
            corr = np.dot(self.A - np.eye(self.A.shape[0]), r)
            A_hat[:-1, -1] = corr
            A_hat[-1, -1] = 1
            B_hat[-1, -1] = 1e-6 # conditionality
            if self.N:
                K, S, E = control.lqr(A_hat, B_hat, self.Q, self.R, self.N)
            else:
                K, S, E = control.lqr(A_hat, B_hat, self.Q, self.R)
            self.K[i] = K[:, :-1].astype(self.dtype)
            self.S[i] = S[:-1, :-1].astype(self.dtype)
        # self.S = S.astype(self.dtype)
        # self.E = E


    @property
    def dim(self):
        return int(self.K.shape[1])

    def get_u(self, x_ref, x, dt):
        u = np.zeros((self.K.shape[0], self.K.shape[1]), dtype=self.dtype)
        for i, k in enumerate(self.K):
            u[i, :] = np.dot(k, x[i] - x_ref[i])
        return u

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