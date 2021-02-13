import tensorflow as tf
from config.BaseConfig import BaseConfig
from . import BaseEvaluator
import numpy as np
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt

class MultiClassificationEvaluator(BaseEvaluator):

    def __init__(self, config: BaseConfig, model: tf.keras.Model, data_loader):
        super().__init__(config, model, data_loader)

    def eval(self):
        test_iter = self.data_loader.test_iter
        predict_result = []
        true_result = []
        for idx in range(len(test_iter)):
            data, label = test_iter[idx]
            predict = self.model.predict(data)
            label = np.argmax(label, axis=-1)
            predict = np.argmax(predict, axis=-1)
            predict_result.extend(predict.tolist())
            true_result.extend(label.tolist())
        result = confusion_matrix(true_result, predict_result)
        with open(f'./exp_out/{self.config.name}/eval/result.txt', 'w') as f:
            class_length = map(str, range(len(result[0])))
            f.write('\t' + '\t'.join(class_length) + '\n')
            for idx, r in enumerate(result):
                f.write(f'{idx}\t' + '\t'.join(map(str, r)) + '\n')
            f.flush()

        self.plot_confusion_matrix(result, target_names=[str(i) for i in range(len(result[0]))], normalize=False, save_path=f'./exp_out/{self.config.name}/eval/confusion_matrix.png')

    @staticmethod
    def plot_confusion_matrix(cm,
                              target_names,
                              title='Confusion matrix',
                              cmap=None,
                              normalize=True,
                              save_path=None):
        import itertools

        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        if save_path is not None:
            plt.savefig(save_path)