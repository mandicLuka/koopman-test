from dmd_network import MishmashNetwork, CoordinateTransformNetwork
from data_window_generator import WindowGenerator
from losses import SequenceSquareLoss
import os, yaml, pickle
import tensorflow as tf
import numpy as np
import itertools


_MODEL_REGISTRY = {
    "mishmash" : MishmashNetwork,
    "ctn": CoordinateTransformNetwork
}

_LOSS_REGISTRY = {
    "ss" : SequenceSquareLoss,
    "loss2": CoordinateTransformNetwork
}


def get_dataset(name):
    pass

def create_model(
    model_architecture,
    input_shape,
    optimizer="adam",
    loss="ss",
    run_eagerly=True,
    model_name="new_model",
    loss_params=None,
    **kwargs
    ) -> tf.keras.Model:

    model = _MODEL_REGISTRY[model_architecture](input_shape, 
            model_name=model_name, **kwargs)

    loss = _LOSS_REGISTRY[loss](**loss_params)

    model.compile(optimizer=optimizer, loss=loss, run_eagerly=run_eagerly)
    return model


def load_and_check_config() -> dict:
    with open(os.path.join("config", "config.yml"), "r") as stream:
        #try:
            cfg = yaml.safe_load(stream)
        #except yaml.YAMLError as exc:
        #    print(exc)

    return cfg

def load_dataset(dataset_name):
    with open(os.path.join("datasets", f"{dataset_name}.pkl"), "rb") as stream:
        dataset = pickle.load(stream)

    return list(map(lambda x: x.data, dataset))


def train_models_on_dataset(dataset, train_params:dict):

    input_width = train_params["input_window_width"]
    skip = train_params["input_window_skip"]
    label_width = train_params["input_window_label_width"]

    data = WindowGenerator(input_width, label_width, skip, **train_params) \
            .make_dataset(dataset)

    for example_inputs, _ in data.take(1):
        input_shape = example_inputs.shape[1:]

    for model_name, params in train_params["models"].items():
        model_arch = params["type"]
        model = create_model(model_arch, input_shape,
                model_name=model_name, **params)

        profile = params.get("profile_batch", None)
        if profile:
            tensorboard_callback = \
                tf.keras.callbacks.TensorBoard(log_dir="logs", 
                    histogram_freq=1, profile_batch=profile)
            model.fit(data, epochs=train_params["epochs"], 
                callbacks=[tensorboard_callback])

        model.fit(data, epochs=train_params["epochs"])
    return model
        


def main():
    cfg = load_and_check_config()
    for ds in cfg["datasets"]:
        dataset = load_dataset(ds)
        train_models_on_dataset(dataset, cfg)


if __name__ == "__main__":
    main()