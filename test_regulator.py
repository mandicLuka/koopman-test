from dmd_network import MishmashNetwork, CoordinateTransformNetwork
from regulator import TrackingLQR, PolePlacement, PseudoinverseTracker
from data_loader import load_model
from data_window_generator import WindowGenerator
import tensorflow as tf
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_result(true_trajectories, model, regulator, input_width, dt=0.1, 
        time=10, force_shape=None, plot_dims=None, **kwargs):

    num_forces = force_shape[0]
    ref_trajectories = np.array(list(map(lambda x: x[:, num_forces:], true_trajectories)))
    x0 = np.array(list(map(lambda x: x[:input_width, num_forces:], true_trajectories)))
    # indices = np.argwhere(np.array(trained_mask) < 1).reshape((-1, ))
    # times, output = evolve_forced2_true(model, true_trajectories, input_width, num_forces, time, dt)
    times, output, _ = model.evolve_regulator(regulator, ref_trajectories, x0, time, dt)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")

    plot_dims = plot_dims or [0, 1, 2]
    dim = len(plot_dims)

    w = 10
    ref_trajectories = np.repeat(ref_trajectories[:, w:w+1], ref_trajectories.shape[1], axis=1)

    if dim == 2:
        for traj in ref_trajectories:
            ax.plot(traj[:, plot_dims[0]], traj[:, plot_dims[1]], lw=1.5, linestyle='dashed')
        for traj in output:
            ax.plot(traj[:, plot_dims[0]], traj[:, plot_dims[1]], lw=1)

    if dim >= 3:
        for traj in ref_trajectories:
            ax.plot(traj[:, plot_dims[0]], traj[:, plot_dims[1]], traj[:, plot_dims[2]], lw=1.5, linestyle='dashed')
        for traj in output:
            ax.plot(traj[:, plot_dims[0]], traj[:, plot_dims[1]], traj[:, plot_dims[2]], lw=1)

    # plt.show()


    for i in range(len(ref_trajectories)):
        fig = plt.figure()
        ax = plt.axes()
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_title("Lorenz Attractor")

        for d in plot_dims:
            ax.plot(times, ref_trajectories[i][:len(times), d], lw=1.5, linestyle='dashed')
            ax.plot(times, output[i][:, d], lw=1)

    plt.show()

def test_model_with_regulator_on_dataset(model_name, datasets, test_params:dict) -> tf.keras.Model:
    input_width = test_params["input_window_width"]
    force_shape = test_params.get("force_shape", None)

    model_arch = test_params["type"]
    for trajs in datasets:
        input_shape = ((input_width, trajs[0].shape[-1] - force_shape[0]), force_shape)
        t = (trajs[:1, :input_width, force_shape[0]:], trajs[:1, :1, :force_shape[0]])
        model = load_model(model_arch, input_shape, model_name, test_params, build_with=t)
        regulator = TrackingLQR(model.U.weights[0].numpy())
        # regulator = PolePlacement(model.U.weights[0].numpy(), np.concatenate((np.repeat(-0.5, 16), np.repeat(-1, 16))))
        plot_result(trajs, model, regulator, input_width, **test_params)


def main():
    ds = "matlab_otter"
    model = "matlab_otter"
    test_params = {
        "input_window_width": 1,
        "type": "fctn",
        # "type": "fmishmash",
        "loss": "seq_mse",
        "force_shape": (2, ),
        "dt": 0.2,
        # "time": 30,
        "time": 5,
        "layers": [32, 32, 32],
        "state_layers": [32, 32, 32],
        "force_layers": [32],
        "plot_dims": [0, 1, 2]
        # "plot_dims": [3, 4, 5]
    }
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from data_loader import load_dataset
    # dataset = [load_dataset(ds)[1:2][0]]
    dataset = [load_dataset(ds)[0][0:4]]
    test_model_with_regulator_on_dataset(model, dataset, test_params)


if __name__ == "__main__":
    main()
