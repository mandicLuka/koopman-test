from create_trajectories import TrajectoryDescriptor, Trajectory
import yaml
import os, pickle, json
import numpy as np
from data_encoding import encode_angle_deg
from scipy.io import loadmat


MATLAB_DATA_PATH = "matlab"
MAT_FILE = "otter_2dof_test.mat"
DATASET_NAME = "otter_2dof_test"

## OTTER
STATE_DEF = {
    "PWM": None,
    # "Ang": [
    #     (0, lambda x: 10*x),
    #     (1,  lambda x: 10*x),
    #     (2, lambda x: 10*x)
    # ],
    "AngVel": [
        (0, lambda x: 100*x),
        (1,  lambda x: 100*x),
        (2, lambda x: 100*x)
    ],
    "Vel": [
        (0, lambda x: 10*x),
        (1,  lambda x: 10*x),
        (2, lambda x: 10*x)
    ]
    # "Acc": [0, 1, 2],
}

## REMUS
# STATE_DEF = {
#     "Inputs": None,
#     # "Ang": [0, 1, 2],
#     "State": [
#         (0, lambda x: 100*x),
#         (1,  lambda x: 100*x),
#         (2, lambda x: 100*x),
#         (3, lambda x: 100*x),
#         (4,  lambda x: 100*x),
#         (5, lambda x: 100*x),
#     ]
# }


def get_index(idx, default):
    if isinstance(idx, int):
        return [idx]
    elif idx is None:
        return list(range(default))
    return idx

def get_closest_values(logs, t, curr_count):
    if not logs[0].shape:
        return logs[1], 0
    while curr_count < len(logs[0]) - 2:
        curr_time = logs[0][curr_count]
        next_time = logs[0][curr_count+1]
        if abs(t - curr_time) < abs(t - next_time):
            return logs[1][curr_count], curr_count
        curr_count += 1

    return logs[1][-1], len(logs) - 1

def parse_trajectories(trajs, config:dict, sample_time=None):

    max_len = 0
    max_len_name = None
    for k in config:
        logs = trajs[0][k]

        for i, values in enumerate(logs):
            # shape check for empty arrays
            if values[0].shape and len(values[0]) > max_len:
                max_len = len(values[0])
                max_len_name = (k, i)
                end_time = values[0][-1]


    state_dim = 0
    for k, idx in config.items():
        logs = trajs[0][k]
        idx = get_index(idx, len(logs))
        state_dim += len(idx)


    N = int(end_time / sample_time) + 1 if sample_time else max_len

    parsed = np.zeros((len(trajs), N, state_dim))
    timestamps = np.zeros((N, ))
    for traj_i in range(len(trajs)):
        print(traj_i)
        log_counters = { k : 0 for k in config.keys()}
        start_time = trajs[traj_i][max_len_name[0]][max_len_name[1]][0][0]
        for t in range(N):
            current_time = start_time + t * sample_time if sample_time \
                else trajs[traj_i][max_len_name[0]][max_len_name[1]][0][t]
            state_count = 0
            state = np.zeros((state_dim, ))
            timestamps[t] = current_time
            for k, idx in config.items():

                logs = trajs[traj_i][k]
                idx = get_index(idx, len(logs))

                for i, name in enumerate(idx):
                    value, count = get_closest_values(logs[i], current_time, log_counters[k])
                    log_counters[k] = count

                    transform = None
                    if isinstance(name, tuple):
                        transform = name[1]

                    if transform:
                        value = transform(value)

                    state[state_count] = value
                    state_count += 1

                parsed[traj_i, t] = state
    
    return parsed, timestamps


def load_trajectories():
    if MAT_FILE is not None:
        full_path = os.path.join(MATLAB_DATA_PATH, MAT_FILE)
        files = [full_path]
    else:
        list_dir = os.listdir(MATLAB_DATA_PATH)
        files = [os.path.join(MATLAB_DATA_PATH, f) for f in list_dir]
    for f in files:
        if not f.endswith(".mat"):
            continue
        mat = loadmat(full_path)

        v = mat["exportTrajectories"]

        trajs = []
        for k in range(v.shape[1]):

            parsed = {}
            for name, idx in STATE_DEF.items():

                if name not in v.dtype.names:
                    raise Exception(f"Name {name} does not exist in the dataset")
                
                struct = v[name][0][k]

                idx = get_index(idx, struct.size)

                traj = []
                for i in range(struct.size):
                    if i in idx or any(x for x in idx if x[0] == i):
                        data_struct = struct[0][i]
                        data = np.squeeze(data_struct["Data"])
                        times = np.squeeze(data_struct["Time"])
                        traj.append((times, data))

                parsed[name] = traj
            trajs.append(parsed)
        yield trajs

def create_datasets():
    trajs = []
    for load_traj in load_trajectories():
        parsed, timestamps = parse_trajectories(load_traj, STATE_DEF, 0.2)
        traj = Trajectory(
            TrajectoryDescriptor(parsed[:, 0], timestamps[1] - timestamps[0], 
                timestamps[0], timestamps[-1]),
            parsed, timestamps
        )

        trajs.append(traj)

    with open(os.path.join("datasets", f"{DATASET_NAME}.pkl"), "wb") as stream:
        #try:
            pickle.dump(trajs, stream)
        #except pickle.PickleError as exc:
        #    print(exc)



if __name__ == "__main__":
    create_datasets()
