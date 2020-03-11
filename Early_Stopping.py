import numpy as np
import tensorflow as tf

class early_stopping:
    def __init__(self, patience):
        self.count = 1
        self.patience = patience

    def __call__(self, val_metric, epoch):
        if len(val_metric) > 0:
            if val_metric[epoch] > val_metric[epoch - self.count]:
                count = self.count + 1
                if count > self.patience:
                    self.count = 1
                    return True
            else:
                self.count = 1
                return False

    def validation(self, y_true, y_pred):
        mse = tf.abs(y_true - tf.squeeze(y_pred))

        mean1 = np.mean(mse[:365] / 3)
        mean2 = np.mean(mse[365:2 * 365])
        mean3 = np.mean(mse[2 * 365:])

        return mean1 + mean2 + mean3