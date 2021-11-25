from dmd_network import MishmashNetwork, CoordinateTransformNetwork
from data_window_generator import WindowGenerator
from losses import SequenceSquareLoss
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
        

def train_model_on_dataset(model_name, dataset, train_params:dict) -> tf.keras.Model:
    input_width = train_params["input_window_width"]
    skip = train_params["input_window_skip"]
    label_width = train_params["input_window_label_width"]

    data = WindowGenerator(input_width, label_width, skip, shuffle=True, **train_params) \
            .make_dataset(dataset)

    for example_inputs, _ in data.take(1):
        input_shape = example_inputs.shape[1:]

    model_arch = train_params["type"]
    model = create_model(model_arch, input_shape,
            model_name=model_name, **train_params)

    profile = train_params.get("profile_batch", None)
    if profile:
        tensorboard_callback = \
            tf.keras.callbacks.TensorBoard(log_dir="logs", 
                histogram_freq=1, profile_batch=profile)
        model.fit(data, epochs=train_params["epochs"], 
            callbacks=[tensorboard_callback], verbose=2)

    validation_split = train_params.get("validation_split", 0)
    cardinality = data.cardinality().numpy()
    val_size = int(validation_split * cardinality)
    train_size = int((1 - validation_split) * cardinality)
    
    train_ds = data.take(train_size)    
    val_ds = data.skip(train_size).take(val_size)

    history = model.fit(train_ds, epochs=train_params["epochs"], validation_data=val_ds)
    return model, history


def main():
    ds = "duffing"
    train_params = {
        "input_window_width": 1,
        "input_window_skip": 0,
        "input_window_label_width": 1,
        "batch_size": 10,
        "epochs": 3,
        "save_path": "saved_models",
        "validation_split": 0.2, # 0-1
        "type": "mishmash",
        "loss": "ss",
        "layers": [32, 32, 32],
        "loss_params": {
            "gamma": 1
        }
    }
    from data_loader import load_dataset
    dataset = load_dataset(ds)
    train_model_on_dataset("ctn", dataset, train_params)


if __name__ == "__main__":
    main()