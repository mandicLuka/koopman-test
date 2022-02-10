from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class SequenceModelNetwork(tf.keras.Model):

    def __init__(self, input_shape, predict_horizon=1, autoencoder_loss=None,
        loss_mask=None, reg_l2=0, koopman_l1=0, **kwargs):
        super().__init__()
        self.input_window = input_shape[0]
        self.feature_shape = input_shape[1:]
        self.num_features = np.prod(self.feature_shape)
        self.predict_horizon = predict_horizon
        self.autoencoder_loss = autoencoder_loss
        self.loss_mask = loss_mask
        self.reg_l2 = reg_l2
        self.koopman_l1 = koopman_l1

        # predictions will be passed as input because
        # the loss function can be complex 
        # self.inputs = [layers.Input(input_shape)]
    
    @abstractproperty
    def state_embed_dim(self):
        pass

    @abstractmethod
    def embed(self, x):
        pass

    @abstractmethod
    def propagate(self, embedded):
        pass

    @abstractmethod
    def inverse(self, embed):
        pass
        
    def call(self, x):
        e = self.embed(x)
        koopman = self.propagate(e)
        inverse = self.inverse(koopman)
        return inverse

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def evolve(self, x0, time, dt):
        shape = x0.shape
        times = np.arange(0, time, dt)
        x = np.zeros((shape[0], len(times), *shape[2:]))
        num_inputs = x0.shape[1]
        x[:, :num_inputs] = x0
        t = 0
        state = self.embed(x0)
        for i in range(num_inputs, len(times)):
            t = times[i]
            next_state = self.propagate(state)
            x[:, i] = self.inverse(next_state)
            state = next_state
        return times, x


    def evolve_true(self, true_trajs, time, dt):
        input_width = self.input_window

        times = np.arange(0, time, dt)
        x0 = np.array(list(map(lambda x: x[:input_width], true_trajs)))
        x = np.zeros((x0.shape[0], len(times), self.num_features))
        x[:, :input_width] = x0
        for i in range(input_width, len(times)):
            curr_x = true_trajs[:, i-input_width:i]
            state = self.embed(curr_x)
            next_state = self.propagate(state)
            x[:, i] = self.inverse(next_state)
        return times, x


    def evolve_regulator(self, regulator, x_ref, x0, time, dt):
        input_width = self.input_window

        times = np.arange(0, time, dt)
        x = np.zeros((x0.shape[0], len(times), self.num_features))
        x[:, :input_width] = x0
        state = self.embed(x0)
        for i in range(input_width, len(times)):
            current_x_ref = x_ref[:, i]
            x_ref_hat = self.embed(current_x_ref)

            u = regulator.get_u(x_ref_hat, state, dt)
            next_state = self.propagate(state) + u
            x[:, i+1] = self.inverse(next_state)
            state = next_state
        return times, x

    def train_step(self, data):

        data, labels = data

        with tf.GradientTape() as tape:
            state = self.embed(data)
            restored_state = self.inverse(state)
            # labels_reshaped = tf.reshape(labels, (tf.shape(labels)[0] * tf.shape(labels)[1], -1))
            # restored_labels = self.inverse(self.embed(labels_reshaped))
            preds = []
            labels_embed = []
            lifted_states = []
            for i in range(self.predict_horizon):
                next_state = self.propagate(state)
                inv = self.inverse(next_state)
                labels_embed.append(self.embed(labels[:, i]))
                lifted_states.append(next_state)
                preds.append(inv)
                state = next_state

            preds = tf.stack(preds, axis=1)
            labels_embed = tf.stack(labels_embed, axis=1)
            lifted_states = tf.stack(lifted_states, axis=1)

            if self.loss_mask is not None:
                labels = tf.boolean_mask(labels, self.loss_mask, axis=labels.shape[-1])
                preds = tf.boolean_mask(preds, self.loss_mask, axis=preds.shape[-1])

            if self.autoencoder_loss:
                # Updates the metrics tracking the loss
                loss = self.compiled_loss((labels, labels_embed, data[:, -1]), (preds, lifted_states, restored_state))
            else:
                loss = self.compiled_loss((labels_embed, ), (lifted_states, ))

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        if self.autoencoder_loss:
            self.compiled_metrics.update_state((labels, labels_embed, data[:, -1]), (preds, lifted_states, restored_state))
        else:
            self.compiled_metrics.update_state((labels_embed, ), (lifted_states, ))
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        data, labels = data

        state = self.embed(data)
        restored_state = self.inverse(state)
        preds = []
        labels_embed = []
        lifted_states = []
        for i in range(self.predict_horizon):
            next_state = self.propagate(state)
            inv = self.inverse(next_state)
            labels_embed.append(self.embed(labels[:, i]))
            lifted_states.append(next_state)
            preds.append(inv)
            state = next_state

        preds = tf.stack(preds, axis=1)
        labels_embed = tf.stack(labels_embed, axis=1)
        lifted_states = tf.stack(lifted_states, axis=1)

        if self.loss_mask is not None:
            labels = tf.boolean_mask(labels, self.loss_mask, axis=labels.shape[-1])
            preds = tf.boolean_mask(preds, self.loss_mask, axis=preds.shape[-1])

        # Updates the metrics tracking the loss
        if self.autoencoder_loss:
            # Updates the metrics tracking the loss
            self.compiled_loss((labels, labels_embed, data[:, -1]), (preds, lifted_states, restored_state))
            self.compiled_metrics.update_state((labels, labels_embed, data[:, -1]), (preds, lifted_states, restored_state))
        else:
            self.compiled_loss((labels_embed, ), (lifted_states, ))
            self.compiled_metrics.update_state((labels_embed, ), (lifted_states, ))

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


class ForcedSequenceModelNetwork(tf.keras.Model):

    def __init__(self, input_shape, predict_horizon=1, autoencoder_loss=None, 
        loss_mask=None, reg_l2=0, koopman_l1=0, **kwargs):
        super().__init__()

        state_shape, force_shape = input_shape
        self.input_window = state_shape[0]
        self.state_shape = state_shape[1:]
        self.force_shape = force_shape
        self.num_state_features = np.prod(self.state_shape)
        self.num_force_features = np.prod(self.force_shape)
        self.predict_horizon = predict_horizon
        self.autoencoder_loss = autoencoder_loss
        self.loss_mask = loss_mask
        self.reg_l2 = reg_l2
        self.koopman_l1 = koopman_l1

        # predictions will be passed as input because
        # the loss function can be complex 
        # self.inputs = [layers.Input(input_shape)]

    @abstractproperty
    def state_embed_dim(self):
        pass

    @abstractproperty
    def force_embed_dim(self):
        pass
    
    @abstractmethod
    def embed_state(self, x, training=False):
        pass

    @abstractmethod
    def embed_force(self, x):
        pass

    @abstractmethod
    def propagate(self, embedded):
        pass

    @abstractmethod
    def inverse_state(self, embed):
        pass
    
    @abstractmethod
    def inverse_force(self, embed):
        pass
        
    @tf.function
    def call(self, data):
        x, f = data
        e_state = self.embed_state(x)
        e_force = self.embed_force(f)
        koopman = self.propagate((e_state, e_force))
        inverse = self.inverse_state(koopman)
        return inverse

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def evolve_regulator(self, regulator, x_ref, x0, time, dt):
        input_width = self.input_window

        times = np.arange(0, time, dt)
        x = np.zeros((x0.shape[0], len(times), self.num_state_features))
        u = np.zeros((x0.shape[0], len(times), self.num_force_features))
        x[:, :input_width] = x0
        state = self.embed_state(x0)
        current_x_ref = x_ref[:, 10]
        x_ref_hat = self.embed_state(current_x_ref)
        regulator.update_ref(x_ref_hat)
        for i in range(input_width, len(times)):
            # current_x_ref = x_ref[:, i]
            # x_ref_hat = self.embed_state(current_x_ref)
            # regulator.update_ref(x_ref_hat)
            curr_u = regulator.get_u(x_ref_hat, state, dt)
            uu = self.inverse_force(curr_u)
            # uu = np.clip(uu, -1, 1)
            curr_u = self.embed_force(uu)
            next_state = self.propagate((state, curr_u))
            a = next_state - state
            b = x_ref_hat - next_state
            x[:, i] = self.inverse_state(next_state)
            state = next_state
            u[:, i] = self.inverse_force(curr_u)
        return times, x, u

    def evolve_true(self, true_trajs, time, dt):
        
        num_forces = self.num_force_features
        input_width = self.input_window
        x0 = np.array(list(map(lambda x: x[:input_width, num_forces:], true_trajs)))
        forces = np.array(list(map(lambda x: x[:, :num_forces], true_trajs)))
        shape = x0.shape
        times = np.arange(0, time, dt)
        x = np.zeros((shape[0], len(times), *shape[2:]))
        num_prev_inputs = x0.shape[1]
        # x[:, :num_prev_inputs] = x0

        for i in range(num_prev_inputs, len(times)):
            curr_x = true_trajs[:, i-num_prev_inputs:i, num_forces:]
            state_embed = self.embed_state(curr_x)
            force_embed = self.embed_force(forces[:, i])
            new_state = self.propagate((state_embed, force_embed))
            x[:, i] = self.inverse_state(new_state)
        return times, x

    def evolve(self, forces, x0, time, dt):
        num_prev_inputs = self.input_window

        times = np.arange(0, time, dt)
        x = np.zeros((x0.shape[0], len(times), self.num_state_features))
        x[:, :num_prev_inputs] = x0
        state_embed = self.embed_state(x0)
        for i in range(num_prev_inputs, len(times)):
            force_embed = self.embed_force(forces[:, i])
            state_embed = self.propagate((state_embed, force_embed))
            x[:, i] = self.inverse_state(state_embed)
        return times, x

    def train_step(self, data):

        data, labels = data
        (inputs, forces) = data
        (labels, forces_l) = labels

        with tf.GradientTape() as tape:

            state = self.embed_state(inputs)
            force = self.embed_force(forces)
            restored_state = self.inverse_state(state)
            restored_force = self.inverse_force(force)
            preds = []
            labels_embed = []
            lifted_states = []
            for i in range(self.predict_horizon):
                next_state = self.propagate((state, force))
                label_embed = self.embed_state(labels[:, i])
                inv = self.inverse_state(next_state)
                labels_embed.append(label_embed)
                preds.append(inv)
                lifted_states.append(next_state)
                state = next_state
                force = self.embed_force(forces_l[:, i])

            preds = tf.stack(preds, axis=1)
            labels_embed = tf.stack(labels_embed, axis=1)
            lifted_states = tf.stack(lifted_states, axis=1)

            if self.loss_mask is not None:
                labels = tf.boolean_mask(labels, self.loss_mask, axis=labels.shape[-1])
                preds = tf.boolean_mask(preds, self.loss_mask, axis=preds.shape[-1])

            if self.autoencoder_loss:
                # Updates the metrics tracking the loss
                loss = self.compiled_loss((labels, labels_embed, inputs[:, -1], forces[:, -1]), (preds, lifted_states, restored_state, restored_force), regularization_losses=self.losses)
            else:
                loss = self.compiled_loss((labels_embed, ), (lifted_states, ))

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)

        if self.autoencoder_loss:
            self.compiled_metrics.update_state((labels, labels_embed, inputs[:, -1], forces[:, -1]), (preds, lifted_states, restored_state, restored_force))
        else:
            self.compiled_metrics.update_state((labels_embed, ), (lifted_states, ))

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        data, labels = data
        (inputs, forces) = data
        (labels, forces_l) = labels

        state = self.embed_state(inputs)
        force = self.embed_force(forces)
        restored_state = self.inverse_state(state)
        restored_force = self.inverse_force(force)
        preds = []
        labels_embed = []
        lifted_states = []
        for i in range(self.predict_horizon):
            next_state = self.propagate((state, force))
            label_embed = self.embed_state(labels[:, i])
            inv = self.inverse_state(next_state)
            preds.append(inv)
            labels_embed.append(label_embed)
            lifted_states.append(next_state)
            state = next_state
            force = self.embed_force(forces_l[:, i])

        preds = tf.stack(preds, axis=1)
        labels_embed = tf.stack(labels_embed, axis=1)
        lifted_states = tf.stack(lifted_states, axis=1)

        if self.loss_mask is not None:
            labels = tf.boolean_mask(labels, self.loss_mask, axis=labels.shape[-1])
            preds = tf.boolean_mask(preds, self.loss_mask, axis=preds.shape[-1])

        if self.autoencoder_loss:
            # Updates the metrics tracking the loss
            self.compiled_loss((labels, labels_embed, inputs[:, -1], forces[:, -1]), (preds, lifted_states, restored_state, restored_force), regularization_losses=self.losses)
            self.compiled_metrics.update_state((labels, labels_embed, inputs[:, -1], forces[:, -1]), (preds, lifted_states, restored_state, restored_force))
        else:
            self.compiled_loss((labels_embed, ), (lifted_states, ))
            self.compiled_metrics.update_state((labels_embed, ), (lifted_states, ))

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
