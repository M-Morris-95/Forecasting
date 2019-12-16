import pandas as pd
import os
import numpy as np
import datetime as dt
import tensorflow as tf
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

def load_ili_data(path):
    ili_data = pd.read_csv(path, header=None)
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





fold_num = 1
for fold_num in range(1,2):
    data_dir = '/Users/michael/Documents/ili_data/dataset_forecasting_lag28/eng_smoothed_14/fold'+str(fold_num) + '/'

    google_train = load_google_data(data_dir + 'google-train')
    ili_train = load_ili_data(data_dir + 'ili-train')
    y_train = pd.read_csv(data_dir + 'y-train', header=None)


    all_ili = ili_train[:48].append(y_train)
    all_ili = all_ili.reset_index(drop=True)
    stationary = np.asarray(all_ili[1]) - np.concatenate([np.zeros((1)), np.asarray(all_ili[1][:-1])])
    stationary[0] = stationary[1]
    all_ili['stationary']=stationary

    ili_vals = np.asarray(all_ili['stationary'])
    google_train = np.asarray(google_train)

    n = normalizer(google_train, ili_vals)
    google_train, ili_vals = n.normalize(google_train, ili_vals)

    trn_ili = ili_vals[:-21]
    tgt_ili = ili_vals[34:]

    trn_x1 = []
    trn_x2 = []
    trn_y = []
    for i in range(len(trn_ili)-27):
        trn_x1.append(trn_ili[i:i + 28])
        trn_x2.append(google_train[i:i+28])
        trn_y.append(tgt_ili[i:i + 15])

    trn_x1 = np.asarray(trn_x1)
    trn_x1 = trn_x1[:,:,np.newaxis]
    trn_x2 = np.asarray(trn_x2)
    trn_y = np.asarray(trn_y)





#     google_test = load_google_data(data_dir + 'google-test')
#     ili_test = load_ili_data(data_dir + 'ili-test')
#     y_test = pd.read_csv(data_dir + 'y-test', header=None)
#
#     all_ili = ili_test[:48].append(y_test)
#     all_ili = all_ili.reset_index(drop=True)
#
#     stationary = np.asarray(all_ili[1]) - np.concatenate([np.zeros((1)), np.asarray(all_ili[1][:-1])])
#     stationary[0] = stationary[1]
#     all_ili['stationary']=stationary
#
#     ili_vals = np.asarray(all_ili['stationary'])
#     google_test = np.asarray(google_test)
#
#     vals = 0
#     y = []
#     for i in ili_vals:
#         vals+=i
#         y.append(vals)
#
#
#     google_test, ili_vals = n.normalize(google_train, ili_vals)
#
#     tst_ili = ili_vals[:-21]
#     tgt_ili = ili_vals[34:]
#
#     tst_x1 = []
#     tst_x2 = []
#     tst_y = []
#     for i in range(len(tst_ili) - 27):
#         tst_x1.append(tst_ili[i:i + 28])
#         tst_x2.append(google_train[i:i + 28])
#         tst_y.append(tgt_ili[i:i + 15])
#
#     tst_x1 = np.asarray(tst_x1)
#     tst_x1 = tst_x1[:,:,np.newaxis]
#     tst_x2 = np.asarray(tst_x2)
#     tst_y = np.asarray(tst_y)
#
#     print('trn_x1.shape\ttrn_x2.shape\ttrn_y.shape\ttst_x1.shape\ttst_x2.shape\ttst_y.shape')
#     print(trn_x1.shape,'\t', trn_x2.shape,'\t',  trn_y.shape,'\t',  tst_x1.shape,'\t',  tst_x2.shape,'\t',  tst_y.shape)
#
#
#
#
#     # the first branch operates on the first input
#     inputA = Input(shape=[trn_x1.shape[1],1])
#     x = LSTM(1, activation='relu')(inputA)
#     x = Model(inputs=inputA, outputs=x)
#
#
#     # the second branch operates on the first input
#     inputB = Input(shape=trn_x2[1].shape[0:2])
#     y = Conv1D(32, 3)(inputB)
#     y = LSTM(128, activation='relu')(y)
#     y = Model(inputs=inputB, outputs=y)
#
#     # combine the output of the two branches
#     combined = concatenate([x.output, y.output])
#
#     # apply a FC layer and then a regression prediction on the combined outputs
#     z = Dense(128, activation="relu")(combined)
#     z = Dense(15, activation="linear")(z)
#
#     # our model will accept the inputs of the two branches and then output a single value
#     model = Model(inputs=[x.input, y.input], outputs=z)
#
#     optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)
#
#     model.compile(optimizer=optimizer,
#                   loss='mae',
#                   metrics=['mae', 'mse', rmse, pearson])
#
#     model.fit(
#         [trn_x1, trn_x2], trn_y,
#         validation_data=([tst_x1, tst_x2], tst_y),
#         epochs=10, batch_size=8)
#
#     predictions = model.predict([tst_x1, tst_x2])
#     for i in range(15):
#         predictions[:,i] = predictions[:,i] * (n.y_max - n.y_min) + n.y_min
#
#     predictions = np.sum(predictions, axis=1)
#     plt.plot(predictions)
#     plt.show()
#
#
#
#     predictions = model.predict([trn_x1, trn_x2])
#     for i in range(15):
#         predictions[:,i] = predictions[:,i] * (n.y_max - n.y_min) + n.y_min
#
#     predictions = np.sum(predictions, axis=1)
#     for i in range(1, len(predictions)):
#         predictions[i]+=predictions[i-1]
#
#     plt.plot(predictions)
#     plt.show()
#
#
#     predictions = model.predict([tst_x1, tst_x2])
#     for i in range(15):
#         predictions[:,i] = tst_y[:,i] * (n.y_max - n.y_min) + n.y_min
#
#     plt.plot(predictions)
#     plt.show()
#
#     predictions = np.sum(predictions, axis=1)
#     for i in range(1, len(predictions)):
#         predictions[i]+=predictions[i-1]
#
#     plt.plot(predictions)
#     plt.show()
#
#
#     y = []
#     y.append(n.un_normalize(np.sum(tst_y, axis = 1)[0]))
#
#     for i in range(len(tst_y)-1):
#         y.append(y[i]+n.un_normalize(tst_y[i+1][-1]))
#
#     y = np.asarray(y)
#     plt.plot(y)
#     plt.show()
#
#
#
#     # model.fit(x_train, y_train,
#     #           batch_size=16,
#     #           validation_data=(x_test, y_test),
#     #           epochs=50,
#     #           verbose=1)
#     #
#     # model.save_weights('model.hdf5')
#
#
#
#
#
# ########################################################################################################################
# #######################                                 PLOTTING                                 #######################
# ########################################################################################################################
#
# #     save_dir = '/Users/michael/Documents/github/Forecasting/Logging/' + timestamp + '/Fold_'+str(fold_num)
# #     if not os.path.exists(save_dir):
# #         os.makedirs(save_dir)
# #
# #     os.chdir(save_dir)
# #     training_stats = pd.DataFrame(model.history.history)
# #     training_stats.to_csv(r'Fold_'+str(fold_num)+'_training_stats.csv')
# #
# #     plt.figure(1)
# #     plt.subplot(2, 2, fold_num)
# #     plt.plot(training_stats.mae)
# #     plt.plot(training_stats.val_mae)
# #     plt.grid(b=True, which='major', color='#666666', linestyle='-')
# #     plt.minorticks_on()
# #     plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# #
# #
# #     plt.figure(2)
# #     plt.subplot(2, 2, fold_num)
# #     train_pred = model.predict(x_train)
# #     plt.plot(pd.to_datetime(y_train_index), train_pred)
# #     plt.plot(pd.to_datetime(y_train_index), y_train)
# #     plt.legend(['prediction', 'true'])
# #     plt.grid(b=True, which='major', color='#666666', linestyle='-')
# #     plt.minorticks_on()
# #     plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# #
# #
# #     plt.figure(3)
# #     plt.subplot(2, 2, fold_num)
# #     test_pred = model.predict(x_test)
# #     plt.plot(pd.to_datetime(y_test_index), test_pred)
# #     plt.plot(pd.to_datetime(y_test_index), y_test)
# #     plt.gcf().autofmt_xdate()
# #     plt.legend(['prediction', 'true'])
# #     plt.grid(b=True, which='major', color='#666666', linestyle='-')
# #     plt.minorticks_on()
# #     plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# #
# #
# #     if not 'Best_Results' in locals():
# #         temp = pd.DataFrame(training_stats.iloc[np.argmin(training_stats.val_loss.values)])
# #         temp = temp.rename(columns={temp.columns[0]: 'Fold'+ str(fold_num)})
# #         Best_Results = temp
# #     else:
# #         temp = pd.DataFrame(training_stats.iloc[np.argmin(training_stats.val_loss.values)])
# #         temp = temp.rename(columns={temp.columns[0]: 'Fold' + str(fold_num)})
# #         Best_Results = pd.concat([Best_Results,temp], axis=1)
# #
# # os.chdir('/Users/michael/Documents/github/Forecasting/Logging/' + timestamp)
# # Best_Results.to_csv(r'Fold_'+str(fold_num)+'_best_stats.csv')
# #
# # plt.figure(1)
# # plt.savefig('training_stats.png')
# #
# #
# # plt.figure(2)
# # plt.savefig('training_predictions.png')
# #
# #
# # plt.figure(3)
# # plt.savefig('validation_predictions.png')
# #
# # plt.figure(1)
# # plt.show()
# # plt.figure(2)
# # plt.show()
# # plt.figure(3)
# # plt.show()