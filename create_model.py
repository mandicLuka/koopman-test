from dmd_network import MishmashNetwork, CoordinateTransformNetwork,ForcedCoordinateTransformNetwork, ForcedMishmashNetwork
from losses import DummyZeroLoss, SequenceMse
import tensorflow as tf

_MODEL_REGISTRY = {
    "mishmash" : MishmashNetwork,
    "fmishmash": ForcedMishmashNetwork,
    "ctn": CoordinateTransformNetwork,
    "fctn": ForcedCoordinateTransformNetwork
}

_LOSS_REGISTRY = {
    "seq_mse" : SequenceMse,
    "mse" : tf.keras.losses.MeanSquaredError,
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

    lambda1 = loss_params and loss_params.get("lambda1", 0)
    lambda2 = loss_params and loss_params.get("lambda2", 0)
    model = _MODEL_REGISTRY[model_architecture](input_shape,
            model_name=model_name, 
            autoencoder_loss=autoencoder_loss, 
            loss_mask=loss_mask, 
            reg_l2=lambda2,
            koopman_l1=lambda1,
            **kwargs)

    losses = [
        _LOSS_REGISTRY[loss](),
    ]
    if autoencoder_loss:
        losses.append(_LOSS_REGISTRY[loss]())
        losses.append(_LOSS_REGISTRY[autoencoder_loss]())

        if isinstance(input_shape[0], list): # if model is with forces
            losses.append(_LOSS_REGISTRY[autoencoder_loss]())

    alpha = loss_params and loss_params.get("alpha", 1)
    beta = loss_params and loss_params.get("beta", 1)
    gamma = loss_params and loss_params.get("gamma", 1)
    # model loss order is: prediction loss, propagation loss and regularization loss
    model.compile(optimizer=optimizer, run_eagerly=run_eagerly, loss=losses, loss_weights=[alpha, beta, gamma, gamma],
        metrics=[tf.metrics.MeanAbsoluteError()])
    return model