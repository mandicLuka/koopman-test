from data_window_generator import WindowGenerator
from create_model import create_model
import tensorflow as tf
import numpy as np
import os
import itertools


def train_model_on_dataset(model_name, dataset, train_params:dict) -> tf.keras.Model:
    input_width = train_params["input_window_width"]
    skip = train_params["input_window_skip"]
    label_width = train_params["input_window_label_width"]

    data = WindowGenerator(input_width, label_width, skip, shuffle=True, **train_params) \
            .make_dataset(dataset)

    for example_inputs, _ in data.take(1):
        input_shape = example_inputs.shape[1:].as_list()

    model_arch = train_params["type"]
    train_params["predict_horizon"] = label_width
    model = create_model(model_arch, input_shape,
            model_name=model_name, **train_params)

    validation_split = train_params.get("validation_split", 0)
    cardinality = data.cardinality().numpy()
    val_size = int(validation_split * cardinality)
    train_size = int((1 - validation_split) * cardinality)

    train_ds = data.take(train_size)    
    val_ds = data.skip(train_size).take(val_size)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join("checkpoint_save", model_name),
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    profile = train_params.get("profile_batch", None)
    if profile:
        tensorboard_callback = \
            tf.keras.callbacks.TensorBoard(log_dir="logs", 
                histogram_freq=1, profile_batch=profile)

        history = model.fit(train_ds, epochs=train_params["epochs"], 
                callbacks=[tensorboard_callback, model_checkpoint_callback], validation_data=val_ds, verbose=1)


    history = model.fit(train_ds, epochs=train_params["epochs"], 
            callbacks=[model_checkpoint_callback] ,validation_data=val_ds, verbose=1)
    return history


def main():
    ds = "lorentz"
    train_params = {
        "input_window_width": 1,
        "input_window_skip": 0,
        "input_window_label_width": 5,
        "batch_size": 32,
        "epochs": 50,
        "save_path": "saved_models",
        "run_eagerly": False,
        "validation_split": 0.2, # 0-1
        # "type": "mishmash",
        # "profile_batch": 100,
        "type": "ctn",
        "loss": "mse",
        # "loss_mask": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        "autoencoder_loss": "mse",
        "layers": [32, 64, 128],
        # "loss_params": {
        #     "alpha": 200,
        #     "beta": 100
        # }
        # "loss_params": {
        #     "alpha": 0.00001,
        #     "beta": 100
        # }
    }
    from data_loader import load_dataset
    # dataset = [[load_dataset(ds)[0][0]]]
    dataset = load_dataset(ds)
    model_name = "lor"
    history = train_model_on_dataset(model_name, dataset, train_params)
    history.model.save_weights(os.path.join(train_params["save_path"], model_name))


if __name__ == "__main__":
    main()
