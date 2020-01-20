from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dropout, Conv1D, GRU, Attention, Dense, Input, concatenate, Flatten, Layer, LayerNormalization, Embedding, Dropout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import metrics
from Tranformer import MultiHeadAttention
import datetime



import tensorflow as tf


class data_builder():
    def __init__(self, args, fold, look_ahead=14, lag = 28, country = 'eng', day_of_the_year=True):
        self.look_ahead = look_ahead
        self.lag = lag
        self.doty = day_of_the_year

        assert country == 'eng' or country == 'us'
        if not args.Server:
            self.directory = '/Users/michael/Documents/ili_data/dataset_forecasting_lag' + str(
                lag) + '/' + country + '_smoothed_' + str(look_ahead) + '/fold' + str(fold) + '/'
        else:
            self.directory = '/home/mimorris/ili_data/dataset_forecasting_lag' + str(
                lag) + '/' + country + '_smoothed_' + str(look_ahead) + '/fold' + str(fold) + '/'


    def load_ili_data(self, path):
        ili_data = pd.read_csv(path, header = None)
        return ili_data[1]

    def load_google_data(self, path):
        google_data = pd.read_csv(path)
        if self.doty:
            temp = google_data['Unnamed: 0'].values
            google_data['Unnamed: 0'] = np.asarray([datetime.datetime.strptime(val, '%Y-%m-%d').timetuple().tm_yday for val in temp])
        else:
            google_data = google_data.drop(['Unnamed: 0'], axis=1)
        return google_data

    def build(self):
        google_train = self.load_google_data(self.directory + 'google-train')
        google_train['ili'] = self.load_ili_data(self.directory + 'ili-train').values

        google_test = self.load_google_data(self.directory + 'google-test')
        google_test['ili'] = self.load_ili_data(self.directory + 'ili-test').values

        y_train_index = pd.read_csv(self.directory + 'y-train', header=None)[0]
        y_test_index = pd.read_csv(self.directory + 'y-test', header=None)[0]

        y_ahead = self.look_ahead + 7

        temp = pd.read_csv(self.directory + 'ili-train', header=None)[self.lag-1:self.lag+self.look_ahead+6]
        temp = temp.append(pd.read_csv(self.directory + 'y-train', header=None))[1].values

        y_train = np.asarray([np.asarray(temp[i:i + y_ahead]) for i in range(len(temp) - y_ahead)])

        temp = pd.read_csv(self.directory + 'ili-test', header=None)[self.lag-1:self.lag+self.look_ahead+6]
        temp = temp.append(pd.read_csv(self.directory + 'y-test', header=None))[1].values
        y_test = np.asarray([np.asarray(temp[i:i + y_ahead]) for i in range(len(temp) - y_ahead)])

        n = normalizer(google_train, y_train)
        google_train = n.normalize(google_train, y_train)
        google_test = n.normalize(google_test, y_test)

        x_train = np.asarray([google_train[i:i + self.lag].values for i in range(len(google_train) - self.lag + 1)])
        x_test = np.asarray([google_test[i:i + self.lag].values for i in range(len(google_test) - self.lag + 1)])

        assert(x_train.shape[0] == y_train.shape[0] == y_train_index.shape[0])
        assert(x_test.shape[0] == y_test.shape[0] == y_test_index.shape[0])

        return x_train, y_train, y_train_index, x_test, y_test, y_test_index

def build_model(x_train, y_train):
    ili_input = Input(shape=[x_train.shape[1],1])
    x = GRU(28, activation='relu', return_sequences=True)(ili_input)
    x = Model(inputs=ili_input, outputs=x)

    google_input = Input(shape=[x_train.shape[1], x_train.shape[2]-1])
    y = GRU(x_train.shape[2]-1, activation='relu', return_sequences=True)(google_input)
    y = GRU(int(0.5*(x_train.shape[2]-1)), activation='relu', return_sequences=True)(y)
    y = Model(inputs=google_input, outputs=y)

    z = concatenate([x.output, y.output])
    z = GRU(y_train.shape[1], activation='relu',return_sequences=False)(z)


    model = Model(inputs=[x.input, y.input], outputs=z)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)

    model.compile(optimizer=optimizer,
                  loss='mae',
                  metrics=['mae', 'mse', metrics.rmse])

    return model

def build_attention(x_train, y_train, num_heads = 1):
    d_model = x_train.shape[1]

    ili_input = Input(shape=[x_train.shape[1],x_train.shape[2]])
    x = GRU(x_train.shape[1], activation='relu', return_sequences=True)(ili_input)

    x = MultiHeadAttention(d_model, num_heads, name="attention")({
        'query': x,
        'key': x,
        'value': x
    })
    x = GRU(int((x_train.shape[2] - 1)), activation='relu', return_sequences=True)(x)
    y = GRU(int(0.75*(x_train.shape[2]-1)), activation='relu', return_sequences=True)(x)
    z = GRU(y_train.shape[1], activation='relu',return_sequences=False)(y)


    model = Model(inputs=ili_input, outputs=z)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)

    model.compile(optimizer=optimizer,
                  loss='mae',
                  metrics=['mae', 'mse', metrics.rmse])

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