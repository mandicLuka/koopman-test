import dynamic_systems
from inspect import getmembers, isfunction
from create_trajectories import TrajectoryGenerator, TrajectoryDescriptor
import yaml
import os, pickle
import numpy as np

def get_dynamic_system(model, system_params):

    # dt = 0.01
    # ran = (-5, 5)
    # model = args.model
    lam = lambda x : x is dynamic_systems.state_space
    dynamic_system_registry = getmembers(dynamic_systems, lam)[0][1]
    sys = next(x for x in dynamic_system_registry.models if x.name == model)
    sys = dynamic_systems.DiscreteDynamicModel(sys, system_params)
    return sys


def load_and_check_datasets_config() -> dict:
    with open(os.path.join("config", "datasets_config.yml"), "r") as stream:
        #try:
            cfg = yaml.safe_load(stream)
        #except yaml.YAMLError as exc:
        #    print(exc)

    return cfg



def generate_trajectories():

    cfg = load_and_check_datasets_config()
    cfg_datasets = cfg.get("datasets")
    for name, cfg_dataset in cfg_datasets.items():
        sys = get_dynamic_system(cfg_dataset.get("system"), cfg_dataset.get("system_params", {}))
        descriptors = []
        for ep in cfg_dataset["episodes"]:
            x_0 = create_initial_states(ep["x_0"])
            descriptors.append(
                TrajectoryDescriptor(x_0, ep["dt_step"], ep["start_time"], ep["end_time"]))
        generated = TrajectoryGenerator(sys, descriptors).evolve()

        with open(os.path.join("datasets", f"{name}.pkl"), "wb") as stream:
            #try:
                pickle.dump(generated, stream)
            #except pickle.PickleError as exc:
            #    print(exc)
        
def create_initial_states(cfg_initial):
    if cfg_initial["type"] == "uniform_box":
        min, max = np.array(cfg_initial["x_min"]), np.array(cfg_initial["x_max"])
        x_0 = min + (max - min) \
            * np.random.random((cfg_initial["num_trajectories"], len(min)))
        return x_0
    else:
        Exception("Unsupported initial state generator type")


if __name__ == "__main__":
    generate_trajectories()
