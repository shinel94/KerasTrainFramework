import tensorflow as tf
import numpy as np
from config.BaseConfig import BaseConfig


class BaseDataloader:
    def __init__(self, config: BaseConfig):
        self.config = config
        train, valid = tf.keras.datasets.mnist.load_data()
        self.train_iter = self.DataGenerator(train[0], train[1], self.config.batch_size, self.config.shuffle)
        self.valid_iter = self.DataGenerator(valid[0], valid[1], self.config.batch_size, self.config.shuffle)

    class DataGenerator(tf.keras.utils.Sequence):
        def __init__(self, data, label, batch_size, shuffle):
            'Initialization'
            self.data = data
            num = np.unique(label, axis=0)
            num = num.shape[0]
            self.label = np.eye(num)[label]

            self.batch_size = batch_size
            self.shuffle = shuffle
            self.on_epoch_end()

        def __len__(self):
            return self.data.shape[0] // self.batch_size

        def __getitem__(self, idx):

            batch_x = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.label[idx * self.batch_size:(idx + 1) * self.batch_size]

            return batch_x, batch_y

        def on_epoch_end(self):
            if self.shuffle:
                idx = np.arange(self.data.shape[0])
                np.random.shuffle(idx)
                self.data = self.data[idx]
                self.label = self.label[idx]