import numpy as np
import tensorflow as tf

class WindowGenerator():

    def __init__(self, input_width, label_width, 
            shift=0,
            batch_size=32,
            shuffle=False,
            sequence_stride=1, 
            **kwargs):

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sequence_stride = sequence_stride

        self.total_window_size = input_width + shift + label_width

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def _split_window(self, window):
        inputs = window[:, self.input_slice, :]
        labels = window[:, self.labels_slice, :]

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, dataset) -> tf.data.Dataset:

        if not isinstance(dataset, list):
            dataset = [dataset]

        total_ds = None
        for data in dataset:
            ds = tf.data.Dataset.from_generator(self._tf_generator, 
                args=[data],
                output_signature=(
                    tf.TensorSpec(shape=(None, self.input_width, None), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, self.label_width, None), dtype=tf.float32)
                )
            )
            total_ds = total_ds.concatenate(ds) if total_ds else ds
        return total_ds

    def __repr__(self):
        return ''.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}'])

    def _tf_generator(self, data):
        for traj in data:
            ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=traj,
                targets=None,
                sequence_length=self.total_window_size,
                sequence_stride=self.sequence_stride,
                shuffle=self.shuffle,
                batch_size=self.batch_size
            ).map(self._split_window)
            for x in ds:
                yield x
