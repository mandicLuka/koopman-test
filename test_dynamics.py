
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
    x = np.zeros((shape[0], num_steps+1, shape[1]))
    x[:, 0, :] = x0
    for i in range(num_steps):
        current_x = x[:, i, :]

        if issubclass(type(model), tf.keras.Model):
            x[:, i+1, :] = model(current_x)
        else:
            dx, dy, dz = model(current_x[:, 0], current_x[:, 1], current_x[:, 2])
            x[:, i+1, 0] = current_x[:, 0] + dt * dx
            x[:, i+1, 1] = current_x[:, 1] + dt * dy
            x[:, i+1, 2] = current_x[:, 2] + dt * dz
    return x

def main():

    parser = argparse.ArgumentParser()

    model = "lorentz_attractor"

    lam = lambda x : isfunction(x) and x.__name__ == model 

    dynamic_model = getmembers(dynamic_models, lam)[0][1] # there  will be only one


    dt = 0.01
    ran = (-20, 20)
    # long trajs
    time = 12
    num_trajs = 700
    epochs = 10

    x0 = ran[0] + (ran[1] - ran[0]) * np.random.random((num_trajs, 3))

    long_x = evolve(dynamic_model, x0, time, dt)

    x0 = ran[0] + (ran[1] - ran[0]) * np.random.random((num_trajs*10, 3))
    short_x = evolve(dynamic_model, x0, time/10, dt)

    # x = x[:100]
    def loss1(y, y_pred):
        # f, s = y[0], y[1]
        # f_pred, s_pred = y_pred[0], y_pred[1]

        return (y - y_pred)**2

    def loss2(y, y_pred):
        # f, s = y[0], y[1]
        # f_pred, s_pred = y_pred[0], y_pred[1]

        return 100*(y - y_pred)**2

    x = long_x[:, :-1, :].reshape((-1, 3))
    x = np.vstack((x, short_x[:, :-1, :].reshape((-1, 3))))
    x_p = long_x[:, 1:, :].reshape((-1, 3))
    x_p = np.vstack((x_p, short_x[:, 1:, :].reshape((-1, 3))))
    # x = data[:, :-1, :].reshape((-1, 3))
    # x_p = data[:, 1:, :].reshape((-1, 3))

    shuffle_indices = np.random.permutation(range(x.shape[0]))

    x = x[shuffle_indices]
    x_p = x_p[shuffle_indices]

    model = CoordinateTransformNetwork(3)
    model.compute_output_shape((None, 3))
    model.compile(optimizer='adam', loss=[loss1, loss2])
    model.fit(x, x_p, epochs=epochs, validation_split=0.2)
    tf.saved_model.save(model, "natrenkani")
    ## test


    time = 10
    x0 = ran[0] + (ran[1] - ran[0]) * np.random.random((2, 3))

    x = evolve(dynamic_model, x0, time, dt)

    output = evolve(model, x0, time, dt)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")


    for traj in x:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=1.5, linestyle='dashed')
    for traj in output:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=1)

    plt.show()


    

    pass


if __name__ == "__main__":
    main()



