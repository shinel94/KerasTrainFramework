import tensorflow as tf
from config.BaseConfig import BaseConfig
from abc import abstractmethod


class BaseEvaluator:

    def __init__(self, config: BaseConfig, model: tf.keras.Model, data_loader):
        self.config = config
        self.model = model
        self.data_loader = data_loader

    @abstractmethod
    def eval(self):
        raise NotImplementedError
