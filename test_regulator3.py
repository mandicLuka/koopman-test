import argparse
import dynamic_models
from inspect import getmembers, isfunction
import numpy as np
import tensorflow as tf
from itertools import product

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dmd_network import CoordinateTransformNetwork
from regulator import PID, PolePlacement


def evolve_regulator(model, regulator, x_ref, x0, time, dt):
    shape = x0.shape
    num_steps = int(time / dt)
    x = np.zeros((shape[0], num_steps+1, *shape[1:]))
    x[:, 0] = x0
    for i in range(num_steps):
        current_x = x[:, i]
        current_x_ref = x_ref[:, i]
        x_hat = model.embed(current_x)
        x_ref_hat = model.embed(current_x_ref)
        u = regulator.get_u(x_ref_hat, x_hat, dt)
        # u_np = u.numpy()
        x_hat_next = model.U(x_hat) + u
        x_next = model.inverse(x_hat_next)
        x[:, i+1] = x_next
    return x

def evolve(model, x0, time, dt):
    shape = x0.shape
    num_steps = int(time / dt)
    x = np.zeros((shape[0], num_steps+1, *shape[1:]))
    x[:, 0] = x0
    t = 0
    for i in range(num_steps):
        current_x = x[:, i]
        t = i * dt
        if issubclass(type(model), tf.keras.Model):
            x[:, i+1] = model(current_x)
        else:
            dx = model(current_x, t)
            x[:, i+1] = current_x + dt * dx
    return x

def plot_result(model, regulator, x0, x_ref, ran, dt):

    time = 10
    dim = model.out_dim
    x = evolve_regulator(model, regulator, x_ref, x0, time, dt)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")

    if dim == 1:
        for traj in x:
            ax.plot(traj[:, 0], lw=1.5, linestyle='dashed')
        for traj in x_ref:
            ax.plot(traj[:, 0], lw=1)

    if dim == 2:
        for traj in x:
            ax.plot(traj[:, 0], traj[:, 1], lw=1.5, linestyle='dashed')
        for traj in x_ref:
            ax.plot(traj[:, 0], traj[:, 1], lw=1)

    if dim >= 3:
        for traj in x:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=1.5, linestyle='dashed')
        for traj in x_ref:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=1)

    plt.show()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train")
    parser.add_argument("--model", default="lorentz_attractor")
    parser.add_argument("--model_name", default="")
    parser.add_argument("--regulator", default="pole_placement")
    args = parser.parse_args()
    args.model_name = args.model_name if args.model_name else args.model + "_model"

    dt = 0.01
    ran = (-20, 20)
    model = args.model
    from dynamic_models import state_space
    lam = lambda x : x is state_space
    dynamic_model_registry = getmembers(dynamic_models, lam)[0][1]
    dynamic_model_type = next(x for x in dynamic_model_registry.models if x.name == model)
    dim, dynamic_model = dynamic_model_type.dim, dynamic_model_type.func


    # x = x[:100]
    def loss1(y, y_pred):
        # f, s = y[0], y[1]
        # f_pred, s_pred = y_pred[0], y_pred[1]
        return (y - y_pred)**2

    def loss2(y, y_pred):
        # f, s = y[0], y[1]
        # f_pred, s_pred = y_pred[0], y_pred[1]

        return 100*(y - y_pred)**2


    custom_objects = { "loss1": loss1, "loss2": loss2, "CoordinateTransformNetwork": CoordinateTransformNetwork }
    model = tf.keras.models.load_model(args.model_name, custom_objects=custom_objects)
    
    time = 10
    x0 = ran[0] + (ran[1] - ran[0]) * np.random.random((1, dim))
    # x_ref = np.zeros((1, int(time / dt) + 1, dim))
    x_ref = evolve(dynamic_model, x0, time, dt)

    print(x0)

    if args.regulator == "pole_placement":
        A = model.U.weights[0].numpy()
        regulator = PolePlacement(A, np.identity(A.shape[0]), [-0.1] * A.shape[0])

    plot_result(model, regulator, x0, x_ref, ran, dt)



if __name__ == "__main__":
    main()



