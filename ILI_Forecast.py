import pandas as pd
import os
import numpy as np
import datetime as dt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Attention, Bidirectional, Flatten
from tensorflow.keras.layers import LSTM, Dropout, Conv1D
from keras_self_attention import SeqSelfAttention

import tensorflow_probability as tfp
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time


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


        # y_val = np.asarray(Y[1])
        # for i in range(y_val.shape[0]):
        #     y_val[i] = (y_val[i]-self.y_min)/(self.y_max - self.y_min)
        # Y_norm = y_val


        return X_norm

    def un_normalize(self, Y):
        y_val = np.asarray(Y[1])
        for i in range(y_val.shape[0]):
            y_val[i] = y_val[i] * (self.y_max - self.y_min) - self.y_min

        Y[1] = y_val
        return Y

import argparse


def GetParser():
    parser = argparse.ArgumentParser(
        description='M-Morris-95 Foprecasting')

    parser.add_argument('--Server', '-S',
                        action = 'store_true',
                        help='is it on the server?')

    return parser

parser = GetParser()
args = parser.parse_args()


def pearson(y_true, y_pred):
    return tfp.stats.correlation(y_true, y_pred)

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true-y_pred)))

def load_google_data(path):
    google_data = pd.read_csv(path)
    google_data = google_data.drop(['Unnamed: 0'], axis=1)
    return google_data

def load_ili_data(path):
    ili_data = pd.read_csv(path, header=None)
    ili_data = ili_data[1]
    return ili_data


def load_data(date_dir):
    google_train = load_google_data(data_dir + 'google-train')
    google_train['ili'] = load_ili_data(data_dir + 'ili-train').values
    leading_days_train = pd.read_csv(data_dir + 'ili-train', header=None)[35:48]
    ili_train = pd.read_csv(data_dir + 'y-train', header=None)
    train_index = ili_train[0]
    ili_train = leading_days_train.append(ili_train, ignore_index=True)

    google_test = load_google_data(data_dir + 'google-test')
    google_test['ili'] = load_ili_data(data_dir + 'ili-test').values
    leading_days_test = pd.read_csv(data_dir + 'ili-test', header=None)[35:48]
    ili_test = pd.read_csv(data_dir + 'y-test', header=None)
    test_index = ili_test[0]
    ili_test = leading_days_test.append(ili_test, ignore_index=True)

    n = normalizer(google_train, ili_train)
    google_train = n.normalize(google_train, ili_train)
    google_test = n.normalize(google_test, ili_test)

    x_train = []
    x_test = []
    y_train = []
    y_test = []
    lag = 28
    forecast = 14

    for i in range(len(google_train) - lag):
        x_train.append(np.asarray(google_train[i:i + lag]))

    for i in range(len(google_test) - lag):
        x_test.append(np.asarray(google_test[i:i + lag]))

    for i in range(len(ili_train) - forecast):
        y_train.append(np.asarray(ili_train[1][i:i + forecast]))

    for i in range(len(ili_test) - forecast):
        y_test.append(np.asarray(ili_test[1][i:i + forecast]))

    x_train, y_train = np.asarray(x_train), np.asarray(y_train)
    x_test, y_test = np.asarray(x_test), np.asarray(y_test)

    return x_train, y_train, x_test, y_test, train_index, test_index

do_plot=True

timestamp = time.strftime('%b-%d-%Y-%H-%M', time.localtime())

plt.figure(1, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
plt.figure(2, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
plt.figure(3, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
results = []


for runs in range(1):
    for fold_num in range(1,5):

        if not args.Server:
            data_dir = '/Users/michael/Documents/ili_data/dataset_forecasting_lag28/eng_smoothed_14/fold'+str(fold_num) + '/'
        else:
            data_dir = '/home/mimorris/ili_data/dataset_forecasting_lag28/eng_smoothed_14/fold'+str(fold_num) + '/'

        x_train, y_train, x_test, y_test, train_index, test_index = load_data(data_dir)

        input = Input(shape = (x_train.shape[1], x_train.shape[2]))
        value = Input(shape = (x_train.shape[1]))
        lstm1 = Bidirectional(LSTM(units=x_train.shape[2], return_sequences=True))(input)
        # attention = SeqSelfAttention(attention_activation='sigmoid')(lstm1)
        attention = Attention(causal=True)([lstm1, output]) #just flipped attention inputs round
        lstm2 = LSTM(units=x_train.shape[2], return_sequences=False, activation='linear')(attention)
        dense = Dense(128, activation='relu')(attention)
        output = Dense(y_train.shape[1], activation = 'linear')(dense)
        model = Model(inputs = input, outputs=output)


        # model = Sequential()
        # model.add(Bidirectional(LSTM(units=128, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2]))))
        # tf.keras.layers.Attention()(
        #     [query_seq_encoding, value_seq_encoding])
        # model.add(Flatten())
        # model.add(Dense(128, activation='relu'))
        # model.add(Dense(y_train.shape[1], activation = 'linear'))

        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)
        model.compile(optimizer=optimizer,
                      loss='mae',
                      metrics=['mae', 'mse', rmse, pearson])

        model.fit(
            x_train, y_train,
            validation_data=([x_test, y_test], y_test),
            epochs=20, batch_size=64)



        model.save_weights('model.hdf5')

        if not args.Server:
            save_dir = '/Users/michael/Documents/github/Forecasting/Logging/' + timestamp + '/Fold_'+str(fold_num)
        else:
            save_dir = '/home/mimorris/Forecasting/Logging' + timestamp + '/Fold_'+str(fold_num)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        os.chdir(save_dir)
        training_stats = pd.DataFrame(model.history.history)
        training_stats.to_csv(r'Fold_'+str(fold_num)+'_training_stats.csv')

        if do_plot:
            plt.figure(1)
            plt.subplot(2, 2, fold_num)
            plt.plot(training_stats.mae)
            plt.plot(training_stats.val_mae)
            plt.grid(b=True, which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


            plt.figure(2)
            plt.subplot(2, 2, fold_num)
            train_pred = model.predict(x_train)[:,13]
            plt.plot(pd.to_datetime(train_index[:-1]), train_pred)
            plt.plot(pd.to_datetime(train_index[:-1]), y_train[:,13])
            plt.legend(['prediction', 'true'])
            plt.grid(b=True, which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


            plt.figure(3)
            plt.subplot(2, 2, fold_num)
            test_pred = model.predict(x_test)[:,13]
            plt.plot(pd.to_datetime(test_index[:-1]), test_pred)
            plt.plot(pd.to_datetime(test_index[:-1]), y_test[:,13])
            plt.gcf().autofmt_xdate()
            plt.legend(['prediction', 'true'])
            plt.grid(b=True, which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


        if not 'Best_Results' in locals():
            temp = pd.DataFrame(training_stats[-1:])
            # temp = temp.rename(columns={temp.columns[0]: 'Fold'+ str(fold_num)})
            Best_Results = temp
        else:
            temp = pd.DataFrame(training_stats[-1:])
            # temp = temp.rename(columns={temp.columns[0]: 'Fold' + str(fold_num)})
            Best_Results = Best_Results.append(temp, ignore_index=True)
        results.append(np.asarray(training_stats[-1:]))

    if not args.Server:
        os.chdir('/Users/michael/Documents/github/Forecasting/Logging/' + timestamp)
    else:
        os.chdir('/home/mimorris/Forecasting/Logging' + timestamp)
    Best_Results = Best_Results.rename(index = {0:'2014/15',1:'2015/16',2:'2016/17',3:'2017/18'})
    Best_Results = Best_Results[['val_mae','val_rmse','val_pearson']]
    Best_Results = Best_Results.rename(columns={'val_mae':'MAE', 'val_rmse':'RMSE','val_pearson':'R'})




Best_Results.to_csv(r'Fold_'+str(fold_num)+'_best_stats.csv')

if do_plot:
    plt.figure(1)
    plt.savefig('training_stats.png')


    plt.figure(2)
    plt.savefig('training_predictions.png')


    plt.figure(3)
    plt.savefig('validation_predictions.png')

    plt.figure(1)
    plt.show()
    plt.figure(2)
    plt.show()
    plt.figure(3)
    plt.show()

# model = Sequential()
# model.add(Conv1D(32, 3, input_shape=x_train[1].shape[0:2]))
# model.add(LSTM(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(1, activation='linear'))
#
#
# optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)
#
# model.compile(optimizer=optimizer,
#               loss='mae',
#               metrics=['mae', 'mse', rmse, pearson])
#
# model.fit(x_train, y_train,
#           batch_size=16,
#           validation_data=(x_test, y_test),
#           epochs=20,
#           verbose=1)