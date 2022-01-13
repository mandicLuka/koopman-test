from create_trajectories import TrajectoryDescriptor, Trajectory
import yaml
import os, pickle, json
import numpy as np
from data_encoding import encode_angle_deg


UNITY_DATA_PATH = "unity"
DATASET_NAME = "unity_ident"
STATE_DEF = {
    "D2/T1": [
        "PwmIn"
    ],
    "D2/T2": [
        "PwmIn"
    ],
    "D2/T3": [
        "PwmIn"
    ],
    "D2/T4": [
        "PwmIn"
    ],
    # "D2/T5": [
    #     "PwmIn"
    # ]
    # "D2/T6": [
    #     "PwmIn"
    # ]
    "Imu/Imu": [
        # "linearAcceleration",
        "angularVelocity",
        ("eulerAngles", encode_angle_deg),
        "localVelocity"
    ]
}


def get_closest_values(logs, t, curr_count):
    while curr_count < len(logs) - 2:
        curr_time = logs[curr_count]["SimulationTime"]
        next_time = logs[curr_count+1]["SimulationTime"]
        if abs(t - curr_time) < abs(t - next_time):
            return logs[curr_count]["Value"], curr_count
        curr_count += 1

    return logs[-1]["Value"], len(logs) - 1

def parse_trajectory(traj, config:dict):
    max_len = 0
    max_len_name = None
    state_dim = 0
    for k, state_names in config.items():
        logs = traj[k]
        first = logs[0]
        if len(logs) > max_len:
            max_len = len(logs)
            max_len_name = k

        for state_name in state_names:
            transform = None
            if isinstance(state_name, tuple):
                transform = state_name[1]
                state_name = state_name[0]

            value = first["Value"][state_name]
            if transform:
                value = transform(value)
            if isinstance(value, list) or \
                    isinstance(value, np.ndarray):
                state_dim += len(value)
            else:
                state_dim += 1

    parsed = np.zeros((max_len, state_dim))
    timestamps = np.zeros((max_len, ))

    log_counters = { k:0 for k in config.keys()}
    for i in range(max_len):
        state_count = 0
        state = np.zeros((state_dim, ))
        current_time = traj[max_len_name][i]["SimulationTime"]
        timestamps[i] = current_time
        for k, state_names in config.items():
            logs = traj[k]
            values, count = get_closest_values(logs, current_time, log_counters[k])
            log_counters[k] = count
            for state_name in state_names:
                # transform value
                transform = None
                if isinstance(state_name, tuple):
                    transform = state_name[1]
                    state_name = state_name[0]

                value = values[state_name]
                if transform:
                    value = transform(value)

                if isinstance(value, list) or \
                        isinstance(value, np.ndarray):
                    new_count = state_count + len(value)
                    state[state_count:new_count] = value
                    state_count = new_count
                else:
                    state[state_count] = value
                    state_count += 1

            parsed[i] = state
    
    return parsed, timestamps



def load_trajectories():
    for f in os.listdir(UNITY_DATA_PATH):
        if not f.endswith(".json"):
            continue
        full_path = os.path.join(UNITY_DATA_PATH, f)
        with open(full_path, "r") as stream:
            j = json.load(stream)
        yield j

def create_datasets():
    trajs = []
    for load_traj in load_trajectories():
        logs = load_traj["Logs"]
        parsed, timestamps = parse_trajectory(logs, STATE_DEF)
        traj = Trajectory(
            TrajectoryDescriptor([parsed[0]], timestamps[1] - timestamps[0], 
                timestamps[0], timestamps[-1]),
            [parsed], timestamps
        )

        trajs.append(traj)

    with open(os.path.join("datasets", f"{DATASET_NAME}.pkl"), "wb") as stream:
        #try:
            pickle.dump(trajs, stream)
        #except pickle.PickleError as exc:
        #    print(exc)



if __name__ == "__main__":
    create_datasets()
