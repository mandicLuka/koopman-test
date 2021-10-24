
import argparse
import dynamic_models
from inspect import getmembers, isfunction
import numpy as np
import tensorflow as tf
from itertools import product

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dmd_network import CoordinateTransformNetwork


def evolve(model, x0, time, dt):
    shape = x0.shape
    num_steps = int(time / dt)
    x = np.zeros((shape[0], num_steps+1, *shape[1:]))
    x[:, 0, :] = x0
    t = 0
    for i in range(num_steps):
        current_x = x[:, i, :]
        t = i * dt
        if issubclass(type(model), tf.keras.Model):
            x[:, i+1] = model(current_x)
        else:
            dx = model(current_x, t)
            x[:, i+1] = current_x + dt * dx
            # x[:, i+1, 1] = current_x[:, 1] + dt * dy
            # x[:, i+1, 2] = current_x[:, 2] + dt * dz
    return x


def plot_result(dynamic_model, model, ran, dt):

    time = 8
    dim = 2
    # dim = model.out_dim
    x0 = ran[0] + (ran[1] - ran[0]) * np.random.random((1, dim))
    x = evolve(dynamic_model, x0, time, dt)

    output = evolve(model, x0, time, dt)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")


    if dim == 1:
        for traj in x:
            ax.plot(traj[:, 0], lw=1.5, linestyle='dashed')
        for traj in output:
            ax.plot(traj[:, 0], lw=1)

    if dim == 2:
        for traj in x:
            ax.plot(traj[:, 0], traj[:, 1], lw=1.5, linestyle='dashed')
        for traj in output:
            ax.plot(traj[:, 0], traj[:, 1], lw=1)

    if dim >= 3:
        for traj in x:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=1.5, linestyle='dashed')
        for traj in output:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=1)

    plt.show()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train")
    parser.add_argument("--model", default="lorentz_attractor")
    parser.add_argument("--model_name", default="")
    args = parser.parse_args()
    args.model_name = args.model_name if args.model_name else args.model + "_model"

    dt = 0.01
    ran = (-5, 5)
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

    if args.mode == "train":
        # long trajs
        time = 12
        num_trajs = 700
        epochs = 10
        validation_split = 0.2

        x0 = ran[0] + (ran[1] - ran[0]) * np.random.random((num_trajs, dim))

        long_x = evolve(dynamic_model, x0, time, dt)

        x0 = ran[0] + (ran[1] - ran[0]) * np.random.random((num_trajs*10, dim))
        short_x = evolve(dynamic_model, x0, time/10, dt)

        x = long_x[:, :-1, :].reshape((-1, dim))
        x = np.vstack((x, short_x[:, :-1, :].reshape((-1, dim))))
        x_p = long_x[:, 1:, :].reshape((-1, dim))
        x_p = np.vstack((x_p, short_x[:, 1:, :].reshape((-1, dim))))
        # x = data[:, :-1, :].reshape((-1, 3))
        # x_p = data[:, 1:, :].reshape((-1, 3))

        shuffle_indices = np.random.permutation(range(x.shape[0]))

        x = x[shuffle_indices]
        x_p = x_p[shuffle_indices]

        # model = CoordinateTransformNetwork(dim)
        from dmd_network import MishmashNetwork
        model = MishmashNetwork(dim)
        model.compile(optimizer='adam', loss=[loss1])
        model.fit([x, x], x_p, epochs=epochs, validation_split=validation_split)
        model.save(args.model_name)
        ## test

    if args.mode == "test":
        custom_objects = { "loss1": loss1, "loss2": loss2 }
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(args.model_name)
            # keras.Model.from_config(config)


    plot_result(dynamic_model, model, ran, dt)



if __name__ == "__main__":
    main()



