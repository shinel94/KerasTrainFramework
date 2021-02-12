from config.BaseConfig import BaseConfig
import tensorflow as tf


def builder(config: BaseConfig):
    callbacks = []
    for callback in config.callbacks:
        callbacks.append(getattr(tf.keras.callbacks, callback['name'])(**callback['parameters']))
    return callbacks
