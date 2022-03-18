from data_loader import load_dataset
import numpy as np
import os
import pydmd
import matplotlib.pyplot as plt
from create_trajectories import Trajectory
import pickle

def main():

    ds = "otter_2dof_test_current"
    extended = True

    # dataset = [[load_dataset(ds)[0][0]]]
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    dataset = load_dataset(ds)

    train_s = dataset[0][1].T
    train_s = encode_states(train_s, extended)
    test_s = dataset[0][0].T
    test_s = encode_states(test_s, extended)

    dmd = pydmd.DMD(20, 20, exact=True, opt=True)
    dmd.fit(train_s)


    plot_dims = [2, 5, 6]
    # plot_dims = [5, 6, 7]
    # time = 4828
    time = 2000
    dt = 0.2

    # step_size = 40
    # step_size += 1
    # predictions = np.zeros((test_s[:, 0].shape[0], time))
    # predictions[:, :step_size] = test_s[:, :step_size]
    # for i in range(step_size, time):
    #     predictions[:, i] = dmd.predict(predictions[:, i-step_size:i-1])[:, 0]
    #     predictions[:4, i] = test_s[:4, i]

    horizon = int(time/dt)
    times = np.zeros(horizon)
    for i in range(1, horizon):
        times[i] = i * dt

    times += 9.95
    predictions = np.real(dmd.predict(test_s)[:, :horizon])
    predictions = decode_states(predictions, extended)
    outputs = predictions.T

    # for i, traj in enumerate(dmd.predict(train_s[:time]).T[plot_dims]):
    for i, traj in enumerate(predictions[plot_dims]):
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(range(time), traj[:time], lw=1.5, linestyle='dashed')
        ax.plot(range(time), test_s.T[:, plot_dims[i]][:time], lw=1)

    plt.show()

    save_test_runs('dmd_test', 'dmd', [times], [outputs], plot_dims)



def save_test_runs(folder_name, model_name, traj_times, traj_outputs, extract_dimensions=[0]):

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    for i, times, outputs in zip(range(len(traj_times)), traj_times, traj_outputs):
        model_path = os.path.join(folder_name, f"{model_name}_run_{i}")
        with open(f"{model_path}.pkl", "wb") as stream:
            pickle.dump(Trajectory(time=times, data=export_data(outputs, extract_dimensions), model_name=model_name), stream)

# TEMP HARDCODED
def export_data(data, extract_dimensions):
    d = data[:, extract_dimensions]
    # d *= 0.01

    d[:, 0] *= 0.01
    d[:, 1] *= 0.1
    d[:, 2] *= 0.1
    return d

def split_dataset(dataset, num_control):
    return dataset
    # return dataset[:, num_control:], dataset[:, :num_control]


def encode_states(dataset, extended):
    if not extended:
        return dataset

    sq = np.multiply(dataset, dataset)
    cub = np.multiply(sq, dataset)
    sin = np.sin(dataset)
    cos = np.cos(dataset)
    
    return np.vstack((dataset, sq, cub, sin, cos))



def decode_states(dataset, extended):
    if not extended:
        return dataset
    
    return dataset[:8, :]

if __name__ == "__main__":
    main()