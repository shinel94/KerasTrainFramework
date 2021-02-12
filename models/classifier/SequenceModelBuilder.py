import tensorflow as tf
from config.BaseConfig import BaseConfig
from . import BaseModelBuilder


class SequenceModelBuilder(BaseModelBuilder):

    def __init__(self, config: BaseConfig):
        self.config = config
        self.model = self.build_model()

    def build_model(self) -> tf.keras.Model:
        input_layer = tf.keras.layers.Input(self.config.input_shape)
        layers = []
        x = input_layer
        layers.append(x)
        for layer in self.config.model_config:
            if layer['name'] == 'Concatenate':
                temp = []
                for index in layer['parameters']['layer_index']:
                    temp.append(layers[index])
                x = tf.keras.layers.Concatenate()(temp)
            else:
                x = getattr(tf.keras.layers, layer['name'])(**layer['parameters'])(x)
            layers.append(x)
        return tf.keras.Model(layers[0], layers[-1])


if __name__ == '__main__':
    from config.BaseConfig import BaseConfig
    import json

    with open(r'F:\HY_Framework\applications\mnist.json', 'r') as f:
        config = BaseConfig.from_json(json.load(f))
    builder = SequenceModelBuilder(config)
    print(builder.model)
