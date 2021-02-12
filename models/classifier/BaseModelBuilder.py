import tensorflow as tf
from abc import ABCMeta, abstractmethod


class BaseModelBuilder(metaclass=ABCMeta):

    @abstractmethod
    def build_model(self) -> tf.keras.models:
        raise NotImplementedError