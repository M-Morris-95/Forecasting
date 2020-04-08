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

    def plot_df(self, logging):
        plt.figure(self.number)
        pred = logging.test_predictions
        cols = pred.columns

        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.grid(b=True, which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.plot(logging.test_ground_truth[cols[0]].values)

        for col in cols:
            if '14/15' in col:
                plt.subplot(2, 2, 1)
                plt.plot(pred[col].values)
            if '15/16' in col:
                plt.subplot(2, 2, 2)
                plt.plot(pred[col].values)
            if '16/17' in col:
                plt.subplot(2, 2, 3)
                plt.plot(pred[col].values)
            if '17/18' in col:
                plt.subplot(2, 2, 4)
                plt.plot(pred[col].values)

        plt.show()

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

    def save(self, name):

        plt.figure(self.number)
        plt.savefig(name)

    def show(self):
        plt.figure(self.number)
        plt.show()

