import tensorflow as tf
from config.BaseConfig import BaseConfig


class BaseTrainer:

    def __init__(self, config: BaseConfig, model: tf.keras.Model, data_loader, callbacks):
        self.config = config
        self.model = model
        self.data_loader = data_loader
        self.callbacks = callbacks

    def train(self):
        self.model.fit(self.data_loader.train_iter,
                       epochs=self.config.epochs,
                       callbacks=self.callbacks,
                       validation_data=self.data_loader.valid_iter)
