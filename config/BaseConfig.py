import os


class BaseConfig:

    def __init__(self, name, mode, trainer_name, evaluator_name,
                 batch_size, shuffle, epochs, input_shape, data_loader_name,
                 models_type, builder_name, model_config, model_compile,
                 callbacks,
                 eval_model_path,
                 *args, **kwargs):
        self.name = name
        self.mode = mode
        self.trainer_name = trainer_name
        self.evaluator_name = evaluator_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epochs = epochs
        self.input_shape = input_shape
        self.data_loader_name = data_loader_name
        self.models_type = models_type
        self.builder_name = builder_name
        self.model_config = model_config
        self.callbacks = callbacks
        self.model_compile = model_compile
        self.eval_model_path = eval_model_path
        self.args = args
        self.kwargs = kwargs
        if self.mode == 'train':
            self.build_train_folder()
        elif self.mode == 'evaluate':
            self.build_eval_folder()
        else:
            pass

    def build_train_folder(self):
        save_path = f'./exp_out/{self.name}'
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(f'{save_path}/models', exist_ok=True)

    def build_eval_folder(self):
        save_path = f'./exp_out/{self.name}/eval'
        os.makedirs(save_path, exist_ok=True)

    @classmethod
    def from_json(cls, json_data: dict):
        return BaseConfig(**json_data)
