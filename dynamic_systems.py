from abc import ABC, abstractmethod, abstractproperty
from typing import Callable
import numpy as np
from functools import wraps

class DynamicSystemFn:

    def __init__(self, model):
        self.name = model.name
        self._func = model.func
        self.dim = model.dim

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

class DiscreteDynamicSystem(ABC):


    @abstractproperty
    def dim(self):
        pass

    @abstractmethod
    def __call__(self, x, dt, t, **kwargs):
        pass

class state_space(object):

    models = []

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, func):
        self.func = func
        self.name = func.__name__
        self.models.append(DynamicSystemFn(self))

class TfLearnedModel(DiscreteDynamicSystem):

    import tensorflow as tf
    def __init__(self, dim, model: tf.keras.Model, transform_input: Callable):
        self._model = model
        self._transform_input = transform_input
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def _prepare_input(self, x, dt, t):
        return self._transform_input(x, dt, t)

    def evolve(self, x, dt, t):
        return self._model(self._prepare_input(x, dt, t))


class DiscreteDynamicModel(DiscreteDynamicSystem):

    def __init__(self, model: DynamicSystemFn, model_params=None):
        self.model_params = model_params or {}
        self.name = model.name
        self._model = model

    @property
    def dim(self):
        return self._model.dim

    def __call__(self, x, dt, t):
        dx = self._model(x, dt, t, **self.model_params)
        return x + dt * dx


@state_space(3)
def lorentz_attractor(x_in, dt, t, *, s=10, r=28, b=2.667):

    x, y, z = x_in[:, 0], x_in[:, 1], x_in[:, 2]
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return np.array((x_dot, y_dot, z_dot)).T

@state_space(2)
def duffing(x_in, dt, t, *, a=0.95, b=0.95, gamma=0.5, delta=1.3, w=1):
    x1, x2 = x_in[:, 0], x_in[:, 1]
    dx1 = x2
    dx2 = - a * x1 - b * np.power(x1, 3) - gamma * x2 + delta * np.cos(w * t)
    return np.array((dx1, dx2)).T