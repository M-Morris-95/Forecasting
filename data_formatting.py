import pandas as pd
import os
import numpy as np
import datetime as dt
import tensorflow as tf
from scipy.interpolate import interp1d
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, LSTM, Dropout, Conv1D


import tensorflow_probability as tfp
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time

class normalizer:
    def __init__(self, x_train, y_train):
        self.x_min = np.min(np.asarray(x_train), axis=0)
        self.x_max = np.max(np.asarray(x_train), axis=0)

        self.y_min = np.min(np.asarray(y_train))
        self.y_max = np.max(np.asarray(y_train))

    def normalize(self, X, Y):

        for i in range(X.shape[0]):
            X[i] = (X[i]-self.x_min)/(self.x_max - self.x_min)

        for i in range(Y.shape[0]):
            Y[i] = (Y[i]-self.y_min)/(self.y_max - self.y_min)



        return X, Y

    def un_normalize(self, Y):
        y_val = np.asarray(Y[1])
        for i in range(y_val.shape[0]):
            y_val[i] = y_val[i] * (self.y_max - self.y_min) - self.y_min

        Y[1] = y_val
        return Y

def load_google_data(path):
    google_data = pd.read_csv(path)
    google_data = google_data.drop(['Unnamed: 0'], axis=1)
    return google_data

def load_ili_data(path, header = None):
    ili_data = pd.read_csv(path, header=header)
    # ili_data = ili_data[1]
    return ili_data

def pearson(y_true, y_pred):
    return tfp.stats.correlation(y_true, y_pred)

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true-y_pred)))


timestamp = time.strftime('%b-%d-%Y-%H-%M', time.localtime())

plt.figure(1, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
plt.figure(2, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
plt.figure(3, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
lag = 28

# ili_train - 3265      2005-08-24 ___ 2005-09-20 ___ 2014-08-02
# +7 days
# google_train - 3265   2005-08-31 ___ 2005-09-27 ___ 2014-08-09
# +14 Days
# y_train - 3238                       2005-10-11 ___ 2014-08-23

# 3259

ILI_data = load_ili_data('/Users/michael/Documents/ili_data/ili_ground_truth/ILI_rates_UK_thursday_linear_interpolation_new.csv', header = 'infer')[:5475]
ILI = np.asarray(ILI_data['weight_ILI'])
new_ili = []
new_idx = []
ili_idx = []
for idx, val in enumerate(ILI):
    if np.mod(idx, 7) == 0:
        new_ili.append(val)
        new_idx.append(idx)
    ili_idx.append(idx)

f = interp1d(np.asarray(new_idx), np.asarray(new_ili), 'cubic')
ILI_data.insert(2, 'cubic_interp',f(np.asarray(ili_idx)))
ILI_data.insert(3, 'cubic_stationary', np.diff(f(np.asarray(ili_idx)[:5476])))


plt.plot(ILI_data['weight_ILI'])



