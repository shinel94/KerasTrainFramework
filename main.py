
def train(config):
    builder = getattr(getattr(__import__("models"), config.models_type), config.builder_name)(config)
    data_loader = getattr(__import__("dataloader"), config.data_loader_name)(config)
    callbacks = getattr(__import__('callbacks'), 'builder')(config)
    builder.model.compile(**config.model_compile)
    trainer = getattr(__import__('trainer'), config.trainer_name)(config, builder.model, data_loader, callbacks)
    trainer.train()

def evaluate(config):
    pass


if __name__ == '__main__':
    from config.BaseConfig import BaseConfig
    import json
    with open('./applications/mnist.json', 'r') as f:
        config = BaseConfig.from_json(json.load(f))

    if config.mode == 'train':
        train(config)
    elif config.mode == 'evaluate':
        evaluate(config)
    else:
        pass
