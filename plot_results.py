import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from create_trajectories import Trajectory
import os
import numpy as np
import pickle
from scipy.io import loadmat

EXPERIMENT_FOLDER = 'experiments'
EXPERIMENT_NAME = 'otter_1dof_1'


def parse_matlab_data(mat):
   return Trajectory(
        time=mat["Time"][33000:35000],
        data=mat["Data"][33000:35000],
        model_name=mat["ModelName"][0]
    )


def mse(x, x_true):
    return np.sqrt(np.sum((x - x_true)**2)) / len(x_true)


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
            data = pickle.load(os.path.join(experiment_path, f))
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
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_title(EXPERIMENT_NAME)

        ax.plot(true_data.time, true_data.data[:, d], lw=1.5, linestyle='dashed')

    for i, data in enumerate(all_data):
        for d in range(data.data.shape[1]):

            plt.figure(plots[d].number)
            ax.plot(data.time, data.data[:, d])
            
            m = mse(data.data[:, d], true_data.data[:, d])
            i, n = max_difference(data.data[:, d], true_data.data[:, d])
            print(f"{data.model_name} score - MSE: {m} INFNORM: {n}")
    plt.show()


if __name__ == "__main__":
    plot()