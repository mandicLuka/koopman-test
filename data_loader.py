import os, yaml, pickle

def load_and_check_config(name:str) -> dict:
    parent = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(parent, "config", f"{name}.yml"), "r") as stream:
        #try:
            cfg = yaml.safe_load(stream)
        #except yaml.YAMLError as exc:
        #    print(exc)

    return cfg

def load_dataset(dataset_name):
    parent = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(parent, "datasets", f"{dataset_name}.pkl"), "rb") as stream:
        dataset = pickle.load(stream)

    return list(map(lambda x: x.data, dataset))