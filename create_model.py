from dmd_network import MishmashNetwork, CoordinateTransformNetwork
from losses import DummyZeroLoss, SequenceMse
import tensorflow as tf

_MODEL_REGISTRY = {
    "mishmash" : MishmashNetwork,
    "ctn": CoordinateTransformNetwork
}

_LOSS_REGISTRY = {
    "mse" : SequenceMse,
    "dummy_zero": DummyZeroLoss
}

def create_model(
    model_architecture,
    input_shape,
    optimizer="adam",
    loss="mse",
    autoencoder_loss=None,
    loss_mask=None,
    run_eagerly=True,
    model_name="new_model",
    loss_params=None,
    **kwargs
    ) -> tf.keras.Model:

    model = _MODEL_REGISTRY[model_architecture](input_shape,
            model_name=model_name, **kwargs)

    losses = [
        _LOSS_REGISTRY[loss](loss_mask=loss_mask)
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