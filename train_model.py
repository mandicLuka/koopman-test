from dmd_network import MishmashNetwork, CoordinateTransformNetwork
from data_window_generator import WindowGenerator
from losses import DummyZeroLoss
import tensorflow as tf
import numpy as np
import itertools


_MODEL_REGISTRY = {
    "mishmash" : MishmashNetwork,
    "ctn": CoordinateTransformNetwork
}

_LOSS_REGISTRY = {
    "mse" : tf.keras.losses.MeanSquaredError,
    "dummy_zero": DummyZeroLoss
}

def create_model(
    model_architecture,
    input_shape,
    output_shape,
    optimizer="adam",
    loss="mse",
    autoencoder_loss=None,
    run_eagerly=True,
    model_name="new_model",
    loss_params=None,
    **kwargs
    ) -> tf.keras.Model:

    model = _MODEL_REGISTRY[model_architecture](input_shape,
            model_name=model_name, **kwargs)

    losses = [
        _LOSS_REGISTRY[loss]()
    ]
    if autoencoder_loss:
        losses.append(_LOSS_REGISTRY[autoencoder_loss]())
    else:
        losses.append(_LOSS_REGISTRY["dummy_zero"]())

    alpha = loss_params and loss_params.get("alpha", None) or 1
    beta = loss_params and loss_params.get("beta", None) or 1
    model.compute_output_shape(input_shape)
    model.compile(optimizer=optimizer, run_eagerly=run_eagerly, loss=losses, loss_weights=[alpha, beta])
    return model

def train_model_on_dataset(model_name, dataset, train_params:dict) -> tf.keras.Model:
    input_width = train_params["input_window_width"]
    skip = train_params["input_window_skip"]
    label_width = train_params["input_window_label_width"]

    data = WindowGenerator(input_width, label_width, skip, shuffle=True, **train_params) \
            .make_dataset(dataset)

    for example_inputs, example_outputs in data.take(1):
        input_shape = example_inputs.shape[1:].as_list()
        output_shape = example_outputs.shape[1:].as_list()

    model_arch = train_params["type"]
    model = create_model(model_arch, input_shape, output_shape,
            model_name=model_name, **train_params)

    validation_split = train_params.get("validation_split", 0)
    cardinality = data.cardinality().numpy()
    val_size = int(validation_split * cardinality)
    train_size = int((1 - validation_split) * cardinality)

    train_ds = data.take(train_size)    
    val_ds = data.skip(train_size).take(val_size)

    profile = train_params.get("profile_batch", None)
    if profile:
        tensorboard_callback = \
            tf.keras.callbacks.TensorBoard(log_dir="logs", 
                histogram_freq=1, profile_batch=profile)

        history = model.fit(train_ds, epochs=train_params["epochs"], 
                callbacks=[tensorboard_callback], validation_data=val_ds, verbose=2)

    history = model.fit(train_ds, epochs=train_params["epochs"], validation_data=val_ds, verbose=2)
    return history


def main():
    ds = "duffing"
    train_params = {
        "input_window_width": 2,
        "input_window_skip": 0,
        "input_window_label_width": 1,
        "batch_size": 10,
        "epochs": 1,
        "save_path": "saved_models",
        "validation_split": 0.2, # 0-1
        "type": "mishmash",
        # "type": "ctn",
        "loss": "mse",
        # "autoencoder_loss": "mse",
        "layers": [32, 32, 32],
        "loss_params": {
            "alpha": 0.00001,
            "beta": 100
        }
    }
    from data_loader import load_dataset
    dataset = load_dataset(ds)
    history = train_model_on_dataset("new_model", dataset, train_params)
    history.model.save("mmm")


if __name__ == "__main__":
    main()
