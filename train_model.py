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
    force_shape = train_params.get("force_shape", None)

    generator = WindowGenerator(input_width, label_width, skip, shuffle=True, **train_params)
    if force_shape:
        data = generator.make_forced_dataset(dataset, force_shape)
        for example_inputs, _ in data.take(1):
            input_shape = (example_inputs[0].shape[1:].as_list(),
                example_inputs[1].shape[1:].as_list())
    else:
        data = generator.make_dataset(dataset)
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

    file_name_sufix = train_params.get("file_name_sufix", "")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join("checkpoint_save", f"{model_name}{file_name_sufix}"),
        save_weights_only=True,
        monitor='val_output_1_mean_absolute_error',
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
            callbacks=[model_checkpoint_callback], validation_data=val_ds, verbose=1)
    return history


def set_hyperparams(train_params, hyperparams):

    import random
    loss_params = {}
    train_params["loss_params"] = loss_params
    for k, v in hyperparams.items():
        loss_params[k] = random.sample(v, 1)[0]
    
def main():
    ds = "otter_2dof_veliki"
    model_name = "otter_m2_15"

    num_runs = 10
    hyperparams = {
        "alpha": [10, 1, 0.1],
        "beta": [1, 1e-1, 1e-2],
        "gamma": [1e-1, 1e-2, 1e-3],
        "lambda1": [1e-4, 1e-5, 1e-6],
        "lambda2": [1e-5, 1e-6, 1e-7]
    }

    # num_runs = 5
    # hyperparams = {
    #     "alpha": [1],
    #     "beta": [5e-1],
    #     "gamma": [1e-2],
    #     "lambda1": [1e-2, 1e-3, 1e-4],
    #     "lambda2": [1e-3, 1e-4, 1e-5]
    # }

    train_params = {
        "input_window_width": 1,
        "input_window_skip": 0,
        "input_window_label_width": 15,
        "force_shape": (2, ),
        "batch_size": 32,
        "epochs": 10,
        "save_path": "saved_models",
        "validation_split": 0.2, # 0-1
        "run_eagerly": False,
        # "type": "fctn",
        "type": "fmishmash",
        # "profile_batch": 100,
        "loss": "seq_mse",
        # "autoencoder_loss": "mse",
        # "loss_mask": [0, 0, 1, 1, 1, 0].
        # "loss_mask": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        # "loss_mask": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        # "loss_mask": [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        # "loss_mask": [0, 1],
        "layers": [32, 32, 32],
        "state_layers": [64, 128, 32],
        "force_layers": [32, 32],
    }
    from data_loader import load_dataset
    # dataset = [[load_dataset(ds)[0][0]]]
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    dataset = load_dataset(ds)

    if train_params.get("autoencoder_loss", None):
        score_name = "val_output_1_mean_absolute_error"
    else:
        score_name = "val_mean_absolute_error"

    best = None
    best_val_loss = float('inf')
    best_i = -1
    best_params = None
    for i in range(num_runs):
        train_params["file_name_sufix"] = i

        set_hyperparams(train_params, hyperparams)

        history = train_model_on_dataset(model_name, dataset, train_params)
        if min(history.history[score_name]) < best_val_loss:
            best = history
            best_val_loss = min(history.history[score_name])
            best_i = i
            best_params = train_params["loss_params"]

    print(f"Best: {best_i}")
    print(best_params)
    history.model.save_weights(os.path.join(train_params["save_path"], model_name))


if __name__ == "__main__":
    main()
