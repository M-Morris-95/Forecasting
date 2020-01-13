from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dropout, Conv1D, GRU, Attention, Dense, Input, concatenate, Flatten, Layer, LayerNormalization, Embedding, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import os
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import argparse

from scipy.stats import pearsonr

import tensorflow as tf

def pearson(y_true, y_pred):
    if type(y_pred) != np.ndarray:
        y_pred = y_pred.numpy()
    y_pred = y_pred.astype('float64')

    # y_true = y_true.squeeze()
    y_true = y_true.astype('float64')
    corr = pearsonr(y_true, y_pred)[0]

    return corr

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true-y_pred)))

def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true-y_pred))

def mae(y_true, y_pred):
    return tf.reduce_mean(tf.math.abs(y_true-y_pred))

def evaluate(y_true, y_pred):
    return mae(y_true, y_pred).numpy(), rmse(y_true, y_pred).numpy(), pearson(y_true, y_pred)

def load_google_data(path):
    google_data = pd.read_csv(path)
    google_data = google_data.drop(['Unnamed: 0'], axis=1)
    return google_data

def load_ili_data(path):
    ili_data = pd.read_csv(path, header=None)
    ili_data = ili_data[1]
    return ili_data

def build_data(fold_dir):
    google_train = load_google_data(fold_dir + 'google-train')
    google_train['ili'] = load_ili_data(fold_dir + 'ili-train').values

    google_test = load_google_data(fold_dir + 'google-test')
    google_test['ili'] = load_ili_data(fold_dir + 'ili-test').values

    # y_train = pd.read_csv(fold_dir + 'y-train', header=None)
    # y_train = np.asarray(y_train[1])
    # y_test = pd.read_csv(fold_dir + 'y-test', header=None)
    # y_test = np.asarray(y_test[1])

    temp = pd.read_csv(fold_dir + 'ili-train', header=None)[27:48]
    temp = temp.append(pd.read_csv(fold_dir + 'y-train', header=None))
    temp = temp[1].values # first of y train is at index 20

    y_train = []
    for i in range(len(temp) - 21):
        y_train.append(np.asarray(temp[i:i + 21]))
    y_train = np.asarray(y_train)

    temp = pd.read_csv(fold_dir + 'ili-test', header=None)[27:48]
    temp = temp.append(pd.read_csv(fold_dir + 'y-test', header=None))
    temp = temp[1].values # first of y test is at index 20

    y_test = []
    for i in range(len(temp) - 21):
        y_test.append(np.asarray(temp[i:i + 21]))
    y_test = np.asarray(y_test)



    n = normalizer(google_train, y_train)
    google_train = n.normalize(google_train, y_train)
    google_test = n.normalize(google_test, y_test)

    x_train = []
    x_test = []
    lag = 28

    for i in range(len(google_train) - lag + 1):
        x_train.append(np.asarray(google_train[i:i + lag]))

    for i in range(len(google_test) - lag + 1):
        x_test.append(np.asarray(google_test[i:i + lag]))

    y_train_index = pd.read_csv(fold_dir + 'y-train', header=None)[0][:len(x_train)]
    x_train = np.asarray(x_train)
    y_train = y_train[:x_train.shape[0]]

    y_test_index = pd.read_csv(fold_dir + 'y-test', header=None)[0][:len(x_test)]
    x_test = np.asarray(x_test)
    y_test = y_test[:x_test.shape[0]]

    return x_train, y_train, y_train_index, x_test, y_test, y_test_index

def build_model(x_train):
    ili_input = Input(shape=[x_train.shape[1],1])
    x = GRU(28, activation='relu', return_sequences=True)(ili_input)
    x = Model(inputs=ili_input, outputs=x)

    google_input = Input(shape=[x_train.shape[1], x_train.shape[2]-1])
    y = GRU(x_train.shape[2]-1, activation='relu', return_sequences=True)(google_input)
    y = GRU(int(0.5*(x_train.shape[2]-1)), activation='relu', return_sequences=True)(y)
    y = Model(inputs=google_input, outputs=y)

    z = concatenate([x.output, y.output])
    z = GRU(21, activation='relu',return_sequences=False)(z)


    model = Model(inputs=[x.input, y.input], outputs=z)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)

    model.compile(optimizer=optimizer,
                  loss='mae',
                  metrics=['mae', 'mse', rmse])

    return model

class normalizer:
    def __init__(self, x_train, y_train):
        self.x_min = np.min(np.asarray(x_train), axis=0)
        self.x_max = np.max(np.asarray(x_train), axis=0)

        self.y_min = np.min(np.asarray(y_train), axis=0)[1]
        self.y_max = np.max(np.asarray(y_train), axis=0)[1]

    def normalize(self, X, Y):
        x_val=np.asarray(X)
        for i in range(x_val.shape[0]):
            x_val[i] = (x_val[i]-self.x_min)/(self.x_max - self.x_min)
        X_norm = pd.DataFrame(data=x_val, columns=X.columns)

        return X_norm

    def un_normalize(self, Y):
        y_val = np.asarray(Y[1])
        for i in range(y_val.shape[0]):
            y_val[i] = y_val[i] * (self.y_max - self.y_min) - self.y_min

        Y[1] = y_val
        return Y

class plotter:
    def __init__(self, number):
        self.number = number
        plt.figure(number, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')

    def plot(self, fold_num, y1, y2, x1 = False):
        plt.figure(self.number)
        plt.subplot(2, 2, fold_num)
        if type(x1) != np.ndarray:
            plt.plot(y1)
            plt.plot(y2)
        else:
            plt.plot(pd.to_datetime(x1), y1)
            plt.plot(pd.to_datetime(x1), y2)
            plt.legend(['prediction', 'true'])

        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    def save(self, name):

        plt.figure(self.number)
        plt.savefig(name)

    def show(self):
        plt.figure(self.number)
        plt.show()