import os, yaml, pickle
import numpy as np
from create_model import create_model

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

    return list((map(lambda x: x.data, dataset)))

def load_model(model_arch, input_shape, model_name, params, build_with=None):

    model = create_model(model_arch, input_shape, 
            model_name=model_name, **params)
    
    model.load_weights(os.path.join(
            "checkpoint_save", model_name
            # "saved_models", model_name
        ))
    if build_with:
        model(build_with)
    return model