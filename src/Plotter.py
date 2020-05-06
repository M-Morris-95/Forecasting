import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class plotter:
    def __init__(self, number, size=[8,6], dpi=200):
        self.number = number
        plt.figure(number, figsize=(size[0], size[1]), dpi=dpi, facecolor='w', edgecolor='k')

    def plot(self, fold_num, y_pred, y_true, x1=False, split=True):
        plt.figure(self.number)
        if split:
            plt.subplot(2, 2, fold_num)
        if type(x1) != np.ndarray:
            plt.plot(y_pred, color="red", label="prediction")
            plt.plot(y_true, color = "blue", label = "ground_truth")

        else:
            plt.plot(pd.to_datetime(x1), y_pred, color="red", label="prediction")
            plt.plot(pd.to_datetime(x1), y_true, color = "blue", label = "ground_truth")

            plt.legend(['prediction', 'true'])

        plt.xlabel('Day of the Season', fontsize=8)
        plt.ylabel('ILI Rate (Infected/100,000)', fontsize=8)
        plt.legend(fontsize=8)
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    def plot_conf(self, fold_num, y_pred, y_true, y_std, split=True):
        if y_std is None:
            self.plot(fold_num, y_pred, y_true)
            return
        plt.figure(self.number)
        if split:
            plt.subplot(2, 2, fold_num)
        if not isinstance(y_pred, (np.ndarray, np.generic) ):
            y_pred = y_pred.numpy()
            y_std = y_std.numpy()
        if y_pred.ndim > 1:
            if y_pred.shape[1] != 0:
                y_pred = y_pred[:, -1]
            else:
                y_pred = np.squeeze(y_pred)

        if y_std.ndim > 1:
            if y_std.shape[1] != 0:
                y_std = y_std[:, -1]
            else:
                y_std = np.squeeze(y_std)

        plt.plot(np.linspace(1, y_pred.shape[0], y_pred.shape[0]), y_pred, color="red", label="prediction")
        plt.plot(np.linspace(1, y_true.shape[0], y_true.shape[0]), y_true, color="blue", label="ground_truth")
        plt.fill_between(np.linspace(1, y_true.shape[0], y_true.shape[0]), np.squeeze(y_pred - y_std), np.squeeze(y_pred + y_std),
                         color="pink", alpha=0.5, label="predict std")
        plt.xlabel('Day of the Season', fontsize=8)
        plt.ylabel('ILI Rate (Infected/100,000)', fontsize=8)
        plt.legend(fontsize=8)
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    def plot_ensemble(self, fold_num, yhats,  y_true):
        plt.figure(self.number)
        plt.subplot(2, 2, fold_num)

        means = []
        for yhat in yhats:
            # plt.scatter(np.linspace(1, y_true.shape[0], y_true.shape[0]), yhat.mean(), s=0.2, alpha=0.5, color="red")
            means.append(yhat)
        means = np.squeeze(np.asarray(means))

        stddev = np.std(means, 0)
        y_pred = np.mean(means, 0)

        plt.fill_between(np.linspace(1, y_true.shape[0], y_true.shape[0]), np.squeeze(y_pred - stddev),
                         np.squeeze(y_pred + stddev),
                         color="pink", alpha=0.5, label="predict std")
        plt.plot(np.linspace(1, y_pred.shape[0], y_pred.shape[0]), y_pred, color="red", label="prediction")
        plt.plot(np.linspace(1, y_true.shape[0], y_true.shape[0]), y_true, color="blue", label="ground_truth")


        plt.xlabel('Day of the Season', fontsize=8)
        plt.ylabel('ILI Rate (Infected/100,000)', fontsize=8)

        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

        return y_pred

    def plot_synthetic(self, x_train, y_train, model, tfd = False):
        if tfd:
            yhats = [model(x_train).mean().numpy() for i in range(25)]
        else:
            yhats = [model(x_train).numpy() for i in range(25)]
        means = np.asarray(yhats)
        means = np.squeeze(np.asarray(means))

        y_std = np.std(means, 0)
        y_pred = np.mean(means, 0)


        plt.figure(self.number)
        plt.plot(x_train, y_pred, color="red", label="train prediction")
        plt.fill_between(x_train[:, 0], np.squeeze(y_pred - y_std), np.squeeze(y_pred + y_std),
                         color="pink", alpha=0.5, label="train predict std")

        plt.scatter(x_train[:, 0], y_train, s=0.3, alpha=0.5, color="mediumslateblue", label="train ground_truth")

        plt.show()

    def plot_loss(self, fold_num, history):
        plt.figure(self.number)
        plt.subplot(2,2,fold_num)
        if 'Likelihood_Loss' in history.columns:
            plt.plot(history.Likelihood_Loss, '-.', color = 'red', label = 'likelihood loss')
            plt.plot(history.KL_Loss, '-', color = 'blue', label = 'KL loss')
            plt.legend()
        else:
            plt.plot(history.Loss, '-', color = 'blue', label = 'KL loss')
        plt.xlabel('Epoch', fontsize=8)
        plt.ylabel('Loss', fontsize=8)
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


    def save(self, name):

        plt.figure(self.number)
        plt.savefig(name)

    def show(self):
        plt.figure(self.number)
        plt.show()

# plt.scatter(np.tile(np.linspace(0, pred.shape[1]-1, pred.shape[1]), (1, pred.shape[0])), pred.reshape(-1), s=0.02, alpha=0.1, color='blue', label='data uncertainty')
# plt.scatter(np.tile(np.linspace(0, pred.shape[1]-1, pred.shape[1]), (1, model_mean.shape[0])), model_mean.reshape(-1), s=0.02, alpha=1,
#             color='red',  label='model uncertainty')
# plt.plot(y_test, color = 'green', label='true value')
# plt.plot(np.mean(model_mean, 0), color='yellow', label='predicted value')
#
# plt.xlim([0, pred.shape[1]])
# plt.ylim([-10,50])
# plt.legend()
# plt.show()

# random_dataset = np.random.rand(np.product(x_test.shape)).reshape(x_test.shape)
# y_pred, y_std, pred, model_mean = model.predict(random_dataset)
# plt.plot(np.linspace(1, y_pred.shape[0], y_pred.shape[0]), y_pred, color="red", label="prediction")
# plt.fill_between(np.linspace(1, y_pred.shape[0], y_pred.shape[0]), np.squeeze(y_pred - y_std), np.squeeze(y_pred + y_std),
#                          color="pink", alpha=0.5, label="predict std")
#
# plt.show()
