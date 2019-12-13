import pandas as pd
import os
import numpy as np
import datetime as dt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import LSTM, Dropout, Conv1D
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

def load_google_data(path):
    google_data = pd.read_csv(path)
    google_data = google_data.drop(['Unnamed: 0'], axis=1)
    return google_data

def load_ili_data(path):
    ili_data = pd.read_csv(path, header=None)
    ili_data = ili_data[1]
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





fold_num = 1
for fold_num in range(1,2):
    data_dir = '/Users/michael/Documents/ili_data/dataset_forecasting_lag28/eng_smoothed_14/fold'+str(fold_num) + '/'

    google_train = load_google_data(data_dir + 'google-train')
    ili_in_train = load_ili_data(data_dir + 'ili-train')
    y_train = pd.read_csv(data_dir + 'y-train', header=None)
    # google_train['ili'] = load_ili_data(data_dir + 'ili-train').values

    all_ili = np.concatenate([ili_in_train[:48], y_train[1]])
    stationary = all_ili - np.concatenate([np.zeros((1)), all_ili[:-1]])
    stationary = stationary[28:]

    

    n = normalizer(google_train, y_train)
    google_train = n.normalize(google_train, y_train)

    google_input = []
    ili_input = []
    ili_target = []

    for i in range(len(google_train)-lag+1):
        google_input.append(np.asarray(google_train[i:i+lag]))
        ili_input.append(np.asarray(google_train[i:i + lag]))

    for i in range(len(y_train)-lag+1):
        google_input.append(np.asarray(google_train[i:i+lag]))


    # google_test = load_google_data(data_dir + 'google-test')
    # ili_in_test = load_ili_data(data_dir + 'ili-test')
    # y_test = pd.read_csv(data_dir + 'y-test', header=None)
    # # google_test['ili'] = load_ili_data(data_dir + 'ili-test').values

    #  x_test = []

    google_test = n.normalize(google_test, y_test)







    # for i in range(len(google_test)-lag+1):
    #     x_test.append(np.asarray(google_test[i:i+lag]))

    y_train_index = y_train[0]
    x_train, y_train = np.asarray(x_train), np.asarray(y_train[1])

    y_test_index = y_test[0]
    x_test, y_test = np.asarray(x_test), np.asarray(y_test[1])



    model = Sequential()
    model.add(Conv1D(32, 3, input_shape=x_train[1].shape[0:2]))
    model.add(LSTM(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)

    model.compile(optimizer=optimizer,
                  loss='mae',
                  metrics=['mae', 'mse', rmse, pearson])



    # model.fit(x_train, y_train,
    #           batch_size=16,
    #           validation_data=(x_test, y_test),
    #           epochs=50,
    #           verbose=1)
    #
    # model.save_weights('model.hdf5')





########################################################################################################################
#######################                                 PLOTTING                                 #######################
########################################################################################################################

#     save_dir = '/Users/michael/Documents/github/Forecasting/Logging/' + timestamp + '/Fold_'+str(fold_num)
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#
#     os.chdir(save_dir)
#     training_stats = pd.DataFrame(model.history.history)
#     training_stats.to_csv(r'Fold_'+str(fold_num)+'_training_stats.csv')
#
#     plt.figure(1)
#     plt.subplot(2, 2, fold_num)
#     plt.plot(training_stats.mae)
#     plt.plot(training_stats.val_mae)
#     plt.grid(b=True, which='major', color='#666666', linestyle='-')
#     plt.minorticks_on()
#     plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#
#
#     plt.figure(2)
#     plt.subplot(2, 2, fold_num)
#     train_pred = model.predict(x_train)
#     plt.plot(pd.to_datetime(y_train_index), train_pred)
#     plt.plot(pd.to_datetime(y_train_index), y_train)
#     plt.legend(['prediction', 'true'])
#     plt.grid(b=True, which='major', color='#666666', linestyle='-')
#     plt.minorticks_on()
#     plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#
#
#     plt.figure(3)
#     plt.subplot(2, 2, fold_num)
#     test_pred = model.predict(x_test)
#     plt.plot(pd.to_datetime(y_test_index), test_pred)
#     plt.plot(pd.to_datetime(y_test_index), y_test)
#     plt.gcf().autofmt_xdate()
#     plt.legend(['prediction', 'true'])
#     plt.grid(b=True, which='major', color='#666666', linestyle='-')
#     plt.minorticks_on()
#     plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#
#
#     if not 'Best_Results' in locals():
#         temp = pd.DataFrame(training_stats.iloc[np.argmin(training_stats.val_loss.values)])
#         temp = temp.rename(columns={temp.columns[0]: 'Fold'+ str(fold_num)})
#         Best_Results = temp
#     else:
#         temp = pd.DataFrame(training_stats.iloc[np.argmin(training_stats.val_loss.values)])
#         temp = temp.rename(columns={temp.columns[0]: 'Fold' + str(fold_num)})
#         Best_Results = pd.concat([Best_Results,temp], axis=1)
#
# os.chdir('/Users/michael/Documents/github/Forecasting/Logging/' + timestamp)
# Best_Results.to_csv(r'Fold_'+str(fold_num)+'_best_stats.csv')
#
# plt.figure(1)
# plt.savefig('training_stats.png')
#
#
# plt.figure(2)
# plt.savefig('training_predictions.png')
#
#
# plt.figure(3)
# plt.savefig('validation_predictions.png')
#
# plt.figure(1)
# plt.show()
# plt.figure(2)
# plt.show()
# plt.figure(3)
# plt.show()