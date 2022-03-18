from dmd_network import MishmashNetwork, CoordinateTransformNetwork
from data_loader import load_model
from data_window_generator import WindowGenerator
from create_trajectories import Trajectory
import tensorflow as tf
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, pickle


## OBSOLETE
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


## OBSOLETE
def evolve_forced(model, forces, x0, time, dt):
    shape = x0.shape
    times = np.arange(0, time, dt)
    x = np.zeros((shape[0], len(times), *shape[2:]))
    num_prev_inputs = x0.shape[1]
    x[:, :num_prev_inputs] = x0
    t = 0
    for i in range(num_prev_inputs, len(times)):
        current_x = x[:, i-num_prev_inputs:i, :]
        # current_x[:, -1, force_indices] = forces[:, i]
        t = i * dt
        if issubclass(type(model), tf.keras.Model):
            next_state = model.predict((current_x, forces[:, i]))
            x[:, i] = next_state
        else:
            dx = model(current_x, t)
            x[:, i] = current_x + dt * dx
            # x[:, i+1, 1] = current_x[:, 1] + dt * dy
            # x[:, i+1, 2] = current_x[:, 2] + dt * dz
    return times, x

def plot_result(true_trajectories, model, input_width, dt=0.1, 
        time=10, force_shape=None, plot_dims=None, start_time=0, **kwargs):

    if force_shape is not None:
        num_forces = force_shape[0]
        state_trajectories = np.array(list(map(lambda x: x[:, num_forces:], true_trajectories)))
        x0 = np.array(list(map(lambda x: x[:input_width, num_forces:], true_trajectories)))
        forces = np.array(list(map(lambda x: x[:, :num_forces], true_trajectories)))
        # times, output = model.evolve_true(true_trajectories, time, dt)
        times, output = model.evolve(forces, x0, time, dt)
    else:
        x0 = np.array(list(map(lambda x: x[:input_width], true_trajectories)))
        # times, output = model.evolve(x0, time, dt)
        times, output = model.evolve_true(true_trajectories, time, dt)
        state_trajectories = true_trajectories

    times = times + start_time
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")

    plot_dims = plot_dims or [0, 1, 2]
    dim = len(plot_dims)

    if dim == 2:
        for traj in state_trajectories:
            ax.plot(traj[:, plot_dims[0]], traj[:, plot_dims[1]], lw=1.5, linestyle='dashed')
        for traj in output:
            ax.plot(traj[:, plot_dims[0]], traj[:, plot_dims[1]], lw=1)

    if dim >= 3:
        for traj in state_trajectories:
            ax.plot(traj[:, plot_dims[0]], traj[:, plot_dims[1]], traj[:, plot_dims[2]], lw=1.5, linestyle='dashed')
        for traj in output:
            ax.plot(traj[:, plot_dims[0]], traj[:, plot_dims[1]], traj[:, plot_dims[2]], lw=1)

    for i in range(len(state_trajectories)):
        fig = plt.figure()
        ax = plt.axes()
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_title("Lorenz Attractor")

        for d in plot_dims:
            ax.plot(times, state_trajectories[i][:len(times), d], lw=1.5, linestyle='dashed')
            ax.plot(times, output[i][:, d], lw=1)

    plt.show()

    return times, output

def test_model_on_dataset(model_name, datasets, test_params:dict) -> tf.keras.Model:
    input_width = test_params["input_window_width"]
    force_shape = test_params.get("force_shape", None)

    model_arch = test_params["type"]

    traj_times = []
    traj_outputs = []
    for trajs in datasets:
        input_shape = (input_width, trajs[0].shape[-1])
        if force_shape:
            input_shape = ((input_width, trajs[0].shape[-1] - force_shape[0]), force_shape)
            t = (trajs[:1, :input_width, force_shape[0]:], trajs[:1, :1, :force_shape[0]])
        else:
            input_shape = (input_width, trajs[0].shape[-1])
            t = trajs[:1, :input_width]

        model = load_model(model_arch, input_shape, model_name, test_params, build_with=t)
        times, output = plot_result(trajs, model, input_width, **test_params)
        traj_times.append(times)
        traj_outputs.append(output)
    
    return traj_times, traj_outputs


# TEMP HARDCODED
def export_data(data, extract_dimensions):
    data[:, 0:3] *= 0.01
    data[:, 3:] *= 0.1
    d = data[:, extract_dimensions]
    return d

    
def save_test_runs(folder_name, model_name, traj_times, traj_outputs, dataset, extract_dimensions):

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    for i, times, outputs, traj in zip(range(len(traj_times)), traj_times, traj_outputs, dataset):
        
        for j, o, d in zip(range(outputs.shape[0]), outputs, traj):
            model_path = os.path.join(folder_name, f"{model_name}_run_{i}_{j}")
            true_path = os.path.join(folder_name, f"true_run_{i}_{j}")
            with open(f"{model_path}.pkl", "wb") as stream:
                pickle.dump(Trajectory(time=times, data=export_data(o, extract_dimensions), model_name=model_name), stream)
            with open(f"{true_path}.pkl", "wb") as stream:
                pickle.dump(Trajectory(time=times, data=export_data(d[:len(times)], extract_dimensions), model_name='true'), stream)

def main():
    # ds = "auv_veliki_full"
    # model = "auv_10"
    ds = "otter_2dof"
    model = "otter_2dof_16"
    save_runs = True
    test_params = {
        "input_window_width": 1,
        "type": "fctn",
        # "type": "fmishmash",
        # "input_state_dependent": True,
        "loss": "seq_mse",
        "autoencoder_loss": "mse",
        "force_shape": (2, ),
        # "trained_mask": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        # "trained_mask": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        # "trained_mask": [0,0 0, 0, 0, 1, 1],
        # "trained_mask": [0, 1],
        "dt": 0.2,
        "time": 200,
        "layers": [32, 32, 32],
        "state_layers": [64, 128, 32],
        "force_layers": [32, 32],
        "extract_dimensions" : [0, 1, 2, 3, 4, 5],
        "start_time": 9.95,
        # "plot_dims": [0, 1, 2, 3, 4, 5]
        # "plot_dims": [0, 1, 2]
        "plot_dims": [3, 4, 5]
    }
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from data_loader import load_dataset
    # dataset = load_dataset(ds)
    # dataset = [load_dataset(ds)[1:2][0]]
    dataset = [load_dataset(ds)[0][5:8]]
    traj_times, traj_outputs = test_model_on_dataset(model, dataset, test_params)

    if save_runs:
        force_shape = test_params["force_shape"]
        save_test_runs("test_runs", model, traj_times, traj_outputs, [x[:, :, force_shape[0]:] for x in dataset], test_params["extract_dimensions"])


if __name__ == "__main__":
    main()
