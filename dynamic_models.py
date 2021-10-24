import numpy as np
from functools import wraps

class state_space(object):

    models = []

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, func):
        self.func = func
        self.name = func.__name__
        self.models.append(self)

@state_space(3)
def lorentz_attractor(x_in, t, s=10, r=28, b=2.667):

    x, y, z = x_in[:, 0], x_in[:, 1], x_in[:, 2]
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return np.array((x_dot, y_dot, z_dot)).T

@state_space(2)
def duffing(x_in, t, a=1, b =-1, d=-0.3, f0=0.5, w=1.2):
    x1, x2 = x_in[:, 0], x_in[:, 1]
    dx1 = x2
    dx2 = a * x1 + b * np.power(x1, 3) + d * x2 + f0 * np.cos(w * t)
    return np.array((dx1, dx2)).T