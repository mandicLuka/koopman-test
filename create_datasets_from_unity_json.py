from create_trajectories import TrajectoryDescriptor, Trajectory
import yaml
import os, pickle, json
import numpy as np
from data_encoding import encode_angle_deg


UNITY_DATA_PATH = "unity"
DATASET_NAME = "unity_ident"
VEHICLES = ["D2", "D2 (1)", "D2 (2)", "D2 (3)"]

STATE_DEF = {
    "T1": [
        "PwmIn"
    ],
    "T2": [
        "PwmIn"
    ],
    "T3": [
        "PwmIn"
    ],
    "T4": [
        "PwmIn"
    ],
    # "D2/T5": [
    #     "PwmIn"
    # ]
    # "D2/T6": [
    #     "PwmIn"
    # ]
    "Imu": [
        # "linearAcceleration",
        "localVelocity",
        "angularVelocity",
        # ("eulerAngles", encode_angle_deg),
        # ("localVelocity", lambda x: [100 * x[0], 100 * x[1]]),
        # ("angularVelocity", lambda x: 100 * x[1]),
        # ("eulerAngles", lambda x: encode_angle_deg(x)[4:]),
    ]
}

def get_topic_name(veh, value):
    return f"{veh}/{value}"


def get_closest_values(logs, t, curr_count):
    while curr_count < len(logs) - 2:
        curr_time = logs[curr_count]["SimulationTime"]
        next_time = logs[curr_count+1]["SimulationTime"]
        if abs(t - curr_time) < abs(t - next_time):
            return logs[curr_count]["Value"], curr_count
        curr_count += 1

    return logs[-1]["Value"], len(logs) - 1

def parse_trajectory(traj, config:dict, sample_time=None):

    end_time = traj["SimulationTime"]
    traj = traj["Logs"]

    first_veh = VEHICLES[0]
    if not sample_time:
        max_len = 0
        max_len_name = None
        for k, state_names in config.items():
            topic_name = get_topic_name(first_veh, k)
            logs = traj[topic_name]
            first = logs[0]
            if len(logs) > max_len:
                max_len = len(logs)
                max_len_name = topic_name

    state_dim = 0
    for k, state_names in config.items():

        topic_name = get_topic_name(first_veh, k)
        logs = traj[topic_name]
        first = logs[0]
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


    N = int(end_time / sample_time) + 1 if sample_time else max_len

    parsed = np.zeros((len(VEHICLES), N, state_dim))
    timestamps = np.zeros((N, ))

    for v_i, veh_name in enumerate(VEHICLES):
        log_counters = { k : 0 for k in config.keys()}
        for i in range(N):
            current_time = i * sample_time if sample_time \
                else traj[max_len_name][i]["SimulationTime"]
            state_count = 0
            state = np.zeros((state_dim, ))
            timestamps[i] = current_time
            for k, state_names in config.items():
                topic_name = get_topic_name(veh_name, k)
                logs = traj[topic_name]
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

                parsed[v_i, i] = state
    
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
        parsed, timestamps = parse_trajectory(load_traj, STATE_DEF, 0.2)
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
