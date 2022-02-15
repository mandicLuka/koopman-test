import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from create_trajectories import Trajectory
import os
import numpy as np
import pickle
from scipy.io import loadmat

EXPERIMENT_FOLDER = 'experiments'
EXPERIMENT_NAME = 'otter_2dof_current'


def parse_matlab_data(mat):
   return Trajectory(
        time=mat["Time"],
        data=mat["Data"],
        model_name=mat["ModelName"][0]
    )


def mse(x, x_true):
    return np.sqrt(np.sum((x - x_true)**2)) / len(x_true)


def mae(x, x_true):
    return np.sum(np.abs(x - x_true)) / len(x_true)

def max_difference(x, x_true):
    abs = np.abs(x - x_true)
    return np.argmax(abs), np.max(abs)

def plot():

    experiment_path = os.path.join(EXPERIMENT_FOLDER, EXPERIMENT_NAME)

    all_data = []
    for f in os.listdir(experiment_path):
        
        if f.endswith(".mat"):
            mat = loadmat(os.path.join(experiment_path, f))
            data = parse_matlab_data(mat)
        elif f.endswith('.pkl'):
            with open(os.path.join(experiment_path, f), 'rb') as stream:
                data = pickle.load(stream)
        else:
            raise Exception('Unknown file extension')
        
        if data.model_name == 'true':
            true_data = data
        else:
            all_data.append(data)


    plots = []
    for d in range(true_data.data.shape[1]):
        plots.append(plt.figure())
        ax = plt.axes()
        ax.legend()
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_title(EXPERIMENT_NAME)
        ax.set(xlim=(0, 300))

        ax.plot(true_data.time, true_data.data[:, d], linestyle='dashed', label='true')


    for i, data in enumerate(all_data):
        for d in range(data.data.shape[1]):

            plt.figure(plots[d].number)
            ax = plt.axes()
            ax.plot(data.time, data.data[:, d], label=data.model_name)

            n = int((data.time[1] - data.time[0]) / (true_data.time[1] - true_data.time[0]))

            indices = np.linspace(0, len(data.time) * n - 1, len(data.time), dtype=np.int32)
            sampled_true_data = np.take(true_data.data[:, d], indices)
            m = mse(data.data[:, d], sampled_true_data)
            i, n = max_difference(data.data[:, d], sampled_true_data)
            o = mae(data.data[:, d], sampled_true_data)
            print(f"{data.model_name}_{d} score - MSE: {m} INFNORM: {n} MAE: {o}")


    for p in plots:
        plt.figure(p.number)
        ax = plt.axes()
        ax.legend()

    plt.show()


if __name__ == "__main__":
    plot()