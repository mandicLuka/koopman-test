from dmd_network import MishmashNetwork, CoordinateTransformNetwork
from data_loader import load_model
from data_window_generator import WindowGenerator
import tensorflow as tf
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def evolve(model, x0, time, dt):
    shape = x0.shape
    times = np.arange(0, time, dt)
    x = np.zeros((shape[0], len(times), *shape[2:]))
    num_inputs = x0.shape[1]
    x[:, :num_inputs] = x0
    t = 0
    for i in range(num_inputs, len(times)):
        current_x = x[:, i-num_inputs:i, :]
        t = times[i]
        if issubclass(type(model), tf.keras.Model):
            next_state = model.predict(current_x)
            x[:, i] = next_state
        else:
            dx = model(current_x, t)
            x[:, i] = current_x + dt * dx
            # x[:, i+1, 1] = current_x[:, 1] + dt * dy
            # x[:, i+1, 2] = current_x[:, 2] + dt * dz
    return times, x

def evolve_forced(model, forces, x0, time, dt):
    shape = x0.shape
    times = np.arange(0, time, dt)
    x = np.zeros((shape[0], len(times), *shape[2:]))
    num_inputs = x0.shape[1]
    x[:, :num_inputs] = x0
    t = 0
    num_forces = forces.shape[2]
    for i in range(num_inputs, len(times)):
        current_x = x[:, i-num_inputs:i, :]
        current_x[:, -1, :num_forces] = forces[:, i]
        t = i * dt
        if issubclass(type(model), tf.keras.Model):
            next_state = model.predict(current_x)
            x[:, i] = next_state
        else:
            dx = model(current_x, t)
            x[:, i] = current_x + dt * dx
            # x[:, i+1, 1] = current_x[:, 1] + dt * dy
            # x[:, i+1, 2] = current_x[:, 2] + dt * dz
    return times, x

def plot_result(true_trajectories, model, input_width, dt=0.1, 
        time=10, forces_count=0, plot_dims=None, **kwargs):

    dim = len(true_trajectories[0][0])
    x0 = np.array(list(map(lambda x: x[:input_width], true_trajectories)))

    if forces_count:
        forces = np.array(list(map(lambda x: x[:, :forces_count], true_trajectories)))
        times, output = evolve_forced(model, forces, x0, time, dt)
    else:
        times, output = evolve(model, x0, time, dt)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")

    plot_dims = plot_dims or [0, 1, 2]
    if dim == 1:
        for traj in true_trajectories:
            ax.plot(traj[:, plot_dims[0]], lw=1.5, linestyle='dashed')
        for traj in output:
            ax.plot(traj[:, plot_dims[0]], lw=1)

    if dim == 2:
        for traj in true_trajectories:
            ax.plot(traj[:, plot_dims[0]], traj[:, plot_dims[1]], lw=1.5, linestyle='dashed')
        for traj in output:
            ax.plot(traj[:, plot_dims[0]], traj[:, plot_dims[1]], lw=1)

    if dim >= 3:
        for traj in true_trajectories:
            ax.plot(traj[:, plot_dims[0]], traj[:, plot_dims[1]], traj[:, plot_dims[2]], lw=1.5, linestyle='dashed')
        for traj in output:
            ax.plot(traj[:, plot_dims[0]], traj[:, plot_dims[1]], traj[:, plot_dims[2]], lw=1)

    # plt.show()


    for i in range(len(true_trajectories)):
        fig = plt.figure()
        ax = plt.axes()
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_title("Lorenz Attractor")

        for d in plot_dims:
            ax.plot(times, true_trajectories[i][:len(times), d], lw=1.5, linestyle='dashed')
            ax.plot(times, output[i][:, d], lw=1)

    plt.show()

def test_model_on_dataset(model_name, datasets, test_params:dict) -> tf.keras.Model:
    input_width = test_params["input_window_width"]
    model_arch = test_params["type"]
    for trajs in datasets:
        input_shape = (input_width, trajs[0].shape[-1])
        model = load_model(model_arch, input_shape, model_name, test_params)
        plot_result(trajs, model, input_width, **test_params)


def main():
    ds = "lorentz"
    test_params = {
        "input_window_width": 4,
        "type": "ctn",
        "loss": "mse",
        "autoencoder_loss": "mse",
        "dt": 0.02,
        "time": 5,
        "layers": [32, 64, 128],
        # "plot_dims": [16, 17, 18]
        # "plot_dims": [10, 11, 12]
    }
    from data_loader import load_dataset
    dataset = [[load_dataset(ds)[0][0]]]
    test_model_on_dataset("lor", dataset, test_params)


if __name__ == "__main__":
    main()
