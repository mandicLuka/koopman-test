import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from create_trajectories import Trajectory
import os
import numpy as np
import pickle
from scipy.io import loadmat

EXPERIMENT_FOLDER = 'experiments'

# EXPERIMENT_NAME = 'otter_1dof'
# X_NAMES = ["Time [s]"]
# Y_NAMES = ["Yaw rate [rad/s]"]
# PLOT_NAMES = ["Yaw model identification"]

# EXPERIMENT_NAME = 'otter_2dof'
# X_NAMES = ["Time [s]"] * 3
# Y_NAMES = ["Yaw rate [rad/s]", "Surge speed [m/s]", "Sway speed [m/s]"]
# PLOT_NAMES = ["Yaw rate", "Surge speed", "Sway speed"]

# EXPERIMENT_NAME = 'otter_2dof_current'
# X_NAMES = ["Time [s]"] * 3
# Y_NAMES = ["Yaw rate [rad/s]", "Surge speed [m/s]", "Sway speed [m/s]"]
# PLOT_NAMES = ["Yaw rate", "Surge speed", "Sway speed"]

EXPERIMENT_NAME = 'auv'
X_NAMES = ["Time [s]"] * 2
Y_NAMES = ["Linear velocities [m/s]", "Angular velocities [rad/s]"]
PLOT_NAMES = ["Linear 6dof", "Angular 6dof"]
PLOT_GROUPS = [[0, 1, 2], [3, 4, 5]]

# EXPERIMENT_NAME = 'otter_15_1'
# X_NAMES = ["Time [s]"] * 6
# Y_NAMES = ["Roll rate [rad/s]", "Pitch rate [rad/s]", "Yaw rate [rad/s]", "Surge speed [m/s]", "Sway speed [m/s]", "Heave speed [m/s]"]
# PLOT_NAMES = ["Roll rate", "Pitch rate", "Yaw rate", "Surge speed", "Sway speed", "Heave speed"]
# # DIM_NAMES = ["Yaw rate", "Surge speed", "Sway speed"]


MODEL_NAMES = {
    "dmd": "EDMD",
    "lsq_1dof": "LSQ",
    "lsq_2dof": "LSQ",
    "grey_1dof": "NLSQ",
    "grey_2dof": "NLSQ",
    "otter2_13": "DK",
    "otter2_13": "DK",
    # "otter_2dof_156": "Koopman15",
    "otter_2dof_16": "DK",
    "otter2_2dof_16": "DK"
}

DIM_NAMES = [
    ["Surge", "Sway", "Heave"],
    ["Roll", "Pitch", "Yaw"],
]

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def parse_matlab_data(mat):
   return Trajectory(
        time=mat["Time"],
        data=mat["Data"],
        model_name=mat["ModelName"][0]
    )


def mse(x, x_true):
    mask = ~np.isnan(x)
    x = x[mask]
    x_true = x_true[mask]
    return np.sqrt(np.sum((x - x_true)**2, axis=0)) / len(x_true)

def mae(x, x_true):
    mask = ~np.isnan(x)
    x = x[mask]
    x_true = x_true[mask]
    return np.sum(np.abs(x - x_true), axis=0) / len(x_true)

def max_difference(x, x_true):
    mask = ~np.isnan(x)
    x = x[mask]
    x_true = x_true[mask]
    abs = np.abs(x - x_true)
    return np.max(abs, axis=0)

def plot():

    experiment_path = os.path.join(EXPERIMENT_FOLDER, EXPERIMENT_NAME)

    all_data = []
    for f in sorted(os.listdir(experiment_path)):
        
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

    np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
    plots = []
    legends = []
    if "PLOT_GROUPS" in globals():
        plot_groups = PLOT_GROUPS
    else:
        plot_groups = [[i] for i in range(true_data.data.shape[1])]

    for i, d in enumerate(plot_groups):
        plots.append(plt.figure(figsize=(6, 3)))
        ax = plt.axes()
        ax.legend()
        ax.set_xlabel(X_NAMES[i])
        ax.set_ylabel(Y_NAMES[i])
        ax.grid()

        ax.plot(true_data.time, true_data.data[:, d], linestyle='dashed')
        if "DIM_NAMES" in globals():
            legends.append(['True {}'.format(x) for x in DIM_NAMES[i]])
        else:
            legends.append(['True'])


    for i, data in enumerate(all_data):
        for p, d in enumerate(plot_groups):

            plt.figure(plots[p].number)
            ax = plt.axes()

            name = MODEL_NAMES.get(data.model_name, data.model_name)

            if "DIM_NAMES" in globals():
                label = DIM_NAMES[p]
                legends[p].extend(DIM_NAMES[p])
            else:
                label = name
                legends[p].extend([name])
            tr = ax.plot(data.time[1:], data.data[1:, d], label=label)

            n = int((data.time[1] - data.time[0]) / (true_data.time[1] - true_data.time[0]))

            max_t = min(len(data.time) * n, len(true_data.time))
            indices = np.linspace(1, max_t - 1, len(data.time) - 1, dtype=np.int32)
            sampled_true_data = np.take(true_data.data[:, d], indices, axis=0)
            m = mse(data.data[1:, d], sampled_true_data)
            n = max_difference(data.data[1:, d], sampled_true_data)
            o = mae(data.data[1:, d], sampled_true_data)
            print(f"{name} score - MSE: {m} MAX: {n} MAE: {o}")


    for i, p in enumerate(plots):
        fig = plt.figure(p.number)
        ax = plt.axes()
        y_offset = 0.08
        box = ax.get_position()
        box.y0 = box.x0 + y_offset
        box.y1 = box.y1 + y_offset
        ax.set_position(box)
        # ax.set(ylim=(-0.2, 0.2))
        # ax.set(ylim=(-1, 2))
        # ax.set(ylim=(-2, 3))
        ax.set(ylim=(-0.15, 0.5))
        ax.set(xlim=(10, 300))
        ax.legend(legends[i], loc='upper right')
        plt.savefig(f"plots/{PLOT_NAMES[i]}.pdf")  

    plt.show()


if __name__ == "__main__":
    plot()