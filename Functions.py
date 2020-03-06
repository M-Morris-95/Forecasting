from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dropout, Conv1D, GRU, Attention, Dense, Input, concatenate, Flatten, Layer, \
    LayerNormalization, Embedding, Dropout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import metrics
from Transformer import MultiHeadAttention
import datetime
import os
import time
import tensorflow as tf


def build_model(x_train, y_train):
    initializer = tf.keras.initializers.glorot_normal(seed=None)
    ili_input = Input(shape=[x_train.shape[1], 1])

    x = GRU(28, activation='relu', return_sequences=True, kernel_initializer=initializer)(ili_input)
    x = Model(inputs=ili_input, outputs=x)

    google_input = Input(shape=[x_train.shape[1], x_train.shape[2] - 1])
    y = GRU(x_train.shape[2] - 1, activation='relu', return_sequences=True, kernel_initializer=initializer)(
        google_input)
    y = GRU(int(0.5 * (x_train.shape[2] - 1)), activation='relu', return_sequences=True,
            kernel_initializer=initializer)(y)
    y = Model(inputs=google_input, outputs=y)

    z = concatenate([x.output, y.output])
    z = GRU(y_train.shape[1], activation='relu', return_sequences=False, kernel_initializer=initializer)(z)

    model = Model(inputs=[x.input, y.input], outputs=z)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)

    model.compile(optimizer=optimizer,
                  loss='mae',
                  metrics=['mae', 'mse', metrics.rmse])

    return model


def simple(x_train):
    ili_input = Input(shape=[x_train.shape[1], x_train.shape[2]])
    flatten = tf.keras.layers.Flatten()(ili_input)
    output = tf.keras.layers.Dense(1)(flatten)
    model = Model(inputs=ili_input, outputs=output)

    return model


def recurrent_attention(x_train, y_train, num_heads=1, regularizer=False):
    if regularizer:
        regularizer = tf.keras.regularizers.l2(0.01)
    else:
        regularizer = None

    d_model = x_train.shape[1]

    ili_input = Input(shape=[x_train.shape[1], x_train.shape[2]])
    x = GRU(x_train.shape[1], activation='relu', return_sequences=True, kernel_regularizer=regularizer)(ili_input)

    x = MultiHeadAttention(d_model, num_heads, name="attention", regularizer=regularizer)({
        'query': x,
        'key': x,
        'value': x
    })
    x = GRU(int((x_train.shape[2] - 1)), activation='relu', return_sequences=True, kernel_regularizer=regularizer)(x)
    y = GRU(int(0.75 * (x_train.shape[2] - 1)), activation='relu', return_sequences=False,
            kernel_regularizer=regularizer)(x)
    y = tf.keras.layers.RepeatVector((y_train.shape[1]))(y)
    z = GRU(1, activation='relu', return_sequences=True, kernel_regularizer=regularizer)(y)
    model = Model(inputs=ili_input, outputs=z)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)

    model.compile(optimizer=optimizer,
                  loss='mae',
                  metrics=['mae', 'mse', metrics.rmse])

    return model


def build_attention(x_train, y_train, num_heads=1, regularizer=False, initializer=None):
    if regularizer:
        regularizer = tf.keras.regularizers.l2(0.01)
    else:
        regularizer = None

    d_model = x_train.shape[1]

    ili_input = Input(shape=[x_train.shape[1], x_train.shape[2]])
    x = GRU(x_train.shape[1], activation='relu', return_sequences=True, kernel_regularizer=regularizer, kernel_initializer=initializer)(ili_input)

    x = MultiHeadAttention(d_model, num_heads, name="attention", regularizer=regularizer, initializer=initializer)({
        'query': x,
        'key': x,
        'value': x
    })
    x = GRU(int((x_train.shape[2] - 1)), activation='relu', return_sequences=True, kernel_regularizer=regularizer, kernel_initializer=initializer)(x)
    y = GRU(int(0.75 * (x_train.shape[2] - 1)), activation='relu', return_sequences=True,
            kernel_regularizer=regularizer, kernel_initializer=initializer)(x)
    z = GRU(y_train.shape[1], activation='relu', return_sequences=False, kernel_regularizer=regularizer, kernel_initializer=initializer)(y)

    model = Model(inputs=ili_input, outputs=z)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)

    model.compile(optimizer=optimizer,
                  loss='mae',
                  metrics=['mae', 'mse', metrics.rmse])

    return model


def simple_GRU(x_train, y_train, regularizer=False):
    if regularizer:
        regularizer = tf.keras.regularizers.l2(0.01)
    else:
        regularizer = None

    ili_input = Input(shape=[x_train.shape[1], x_train.shape[2]])
    x = GRU(x_train.shape[1], activation='relu', return_sequences=True, kernel_regularizer=regularizer)(ili_input)
    x = GRU(int((x_train.shape[2] - 1)), activation='relu', return_sequences=True, kernel_regularizer=regularizer)(x)
    y = GRU(int(0.75 * (x_train.shape[2] - 1)), activation='relu', return_sequences=True,
            kernel_regularizer=regularizer)(x)
    z = GRU(y_train.shape[1], activation='relu', return_sequences=False, kernel_regularizer=regularizer)(y)

    model = Model(inputs=ili_input, outputs=z)

    return model


class data_builder:
    def __init__(self, args, fold, look_ahead=14):
        country = args.Country
        self.look_ahead = look_ahead
        if args.Weather == 'True':
            self.weather = True
        else:
            self.weather = False
        if args.DOTY == 'True':
            self.doty = True
        else:
            self.doty = False

        self.lag = args.Lag

        assert country == 'eng' or country == 'us'
        if not args.Server:
            self.weather_directory = '/Users/michael/Documents/ili_data/Weather/all_weather_data.csv'
            self.directory = '/Users/michael/Documents/ili_data/dataset_forecasting_lag' + str(
                self.lag) + '/' + country + '_smoothed_' + str(look_ahead) + '/fold' + str(fold) + '/'
        else:
            self.weather_directory = '/home/mimorris/ili_data/Weather/all_weather_data.csv'
            self.directory = '/home/mimorris/ili_data/dataset_forecasting_lag' + str(
                self.lag) + '/' + country + '_smoothed_' + str(look_ahead) + '/fold' + str(fold) + '/'

    def load_ili_data(self, path):
        ili_data = pd.read_csv(path, header=None)
        return ili_data[1]

    def load_google_data(self, path):
        google_data = pd.read_csv(path)
        weather = pd.read_csv(self.weather_directory)
        temp = google_data['Unnamed: 0'].values

        for idx, val in enumerate(weather['0']):
            if val == temp[0]:
                weather = weather[idx:]
        for idx, val in enumerate(weather['0']):
            if val == temp[-1]:
                weather = weather[:idx + 1]
        if self.weather:
            weather = weather.reset_index(drop=True)
            google_data['weather mean'] = weather['mean']

        if self.doty:
            google_data['Unnamed: 0'] = np.asarray(
                [datetime.datetime.strptime(val, '%Y-%m-%d').timetuple().tm_yday for val in temp])
        else:
            google_data = google_data.drop(['Unnamed: 0'], axis=1)
        return google_data

    def split(self, x_train, y_train):
        self.years = 3
        self.val_size = 3 * 365
        x_val = x_train[-self.val_size:]
        y_val = y_train[-self.val_size:]
        x_train = x_train[:-self.val_size]
        y_train = y_train[:-self.val_size]
        return x_train, y_train, x_val, y_val

    def build(self, squared, normalise_all=False):
        google_train = self.load_google_data(self.directory + 'google-train')
        google_train['ili'] = self.load_ili_data(self.directory + 'ili-train').values
        # google_train = self.load_ili_data(self.directory + 'ili-train')

        google_test = self.load_google_data(self.directory + 'google-test')
        google_test['ili'] = self.load_ili_data(self.directory + 'ili-test').values
        # google_test = self.load_ili_data(self.directory + 'ili-test')

        y_train_index = pd.read_csv(self.directory + 'y-train', header=None)[0]
        y_test_index = pd.read_csv(self.directory + 'y-test', header=None)[0]

        y_ahead = self.look_ahead + 7

        ili_train = pd.read_csv(self.directory + 'ili-train', header=None)
        y_train = pd.read_csv(self.directory + 'y-train', header=None)
        for idx, val in enumerate(ili_train[0]):
            if val == y_train[0][0]:
                ili_train = ili_train[idx - y_ahead + 1:idx]
        ili_train = ili_train.append(y_train)
        y_train = np.asarray([np.asarray(ili_train[1][i:i + y_ahead]) for i in range(len(ili_train) - y_ahead + 1)])

        ili_test = pd.read_csv(self.directory + 'ili-test', header=None)
        y_test = pd.read_csv(self.directory + 'y-test', header=None)
        for idx, val in enumerate(ili_test[0]):
            if val == y_test[0][0]:
                ili_test = ili_test[idx - y_ahead + 1:idx]
        ili_test = ili_test.append(y_test)
        y_test = np.asarray([np.asarray(ili_test[1][i:i + y_ahead]) for i in range(len(ili_test) - y_ahead + 1)])
        #

        if squared:

            google_train = pd.concat((google_train, google_train.pow(2)), axis=1)
            google_test = pd.concat((google_test, google_test.pow(2)), axis=1)


        if not normalise_all:
            train_ili = google_train.ili
            test_ili = google_test.ili
            google_train = google_train.drop(columns=['ili'])
            google_test = google_test.drop(columns=['ili'])

            n = normalizer(google_train, y_train)
            google_train = n.normalize(google_train, y_train)
            google_test = n.normalize(google_test, y_test)
            google_train = pd.concat((google_train, train_ili), axis=1)
            google_test = pd.concat((google_test, test_ili), axis = 1)

        else:
            n = normalizer(google_train, y_train)
            google_train = n.normalize(google_train, y_train)
            google_test = n.normalize(google_test, y_test)

        self.columns = google_train.columns
        x_train = np.asarray([google_train[i:i + self.lag].values for i in range(len(google_train) - self.lag + 1)])
        x_test = np.asarray([google_test[i:i + self.lag].values for i in range(len(google_test) - self.lag + 1)])

        assert (x_train.shape[0] == y_train.shape[0] == y_train_index.shape[0])
        assert (x_test.shape[0] == y_test.shape[0] == y_test_index.shape[0])

        return x_train, y_train, y_train_index, x_test, y_test, y_test_index


class normalizer:
    def __init__(self, x_train, y_train):
        self.x_min = np.min(np.asarray(x_train), axis=0)
        self.x_max = np.max(np.asarray(x_train), axis=0)

        self.y_min = np.min(np.asarray(y_train), axis=0)[1]
        self.y_max = np.max(np.asarray(y_train), axis=0)[1]

    def normalize(self, X, Y):
        x_val = np.asarray(X)
        for i in range(x_val.shape[0]):
            x_val[i] = (x_val[i] - self.x_min) / (self.x_max - self.x_min)
        X_norm = pd.DataFrame(data=x_val, columns=X.columns)

        return X_norm

    def un_normalize(self, Y):
        y_val = np.asarray(Y[1])
        for i in range(y_val.shape[0]):
            y_val[i] = y_val[i] * (self.y_max - self.y_min) - self.y_min

        Y[1] = y_val
        return Y


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
        plt.figure(self.number)
        if split:
            plt.subplot(2, 2, fold_num)
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

    def save(self, name):

        plt.figure(self.number)
        plt.savefig(name)

    def show(self):
        plt.figure(self.number)
        plt.show()

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

class logger:
    def __init__(self, args):
        timestamp = time.strftime('_%b_%d_%H_%M', time.localtime())
        self.root_directory = os.getcwd()
        self.ret_max_k = args.K
        self.do_stddev = False

        self.save_model = args.Save_Model

        if args.K != 1:
            self.iter = True
        else:
            self.iter = False
        if args.Model == 'ALL':
            self.indvidual_models = True
            self.ret_model = ['GRU', 'ATTENTION', 'ENCODER']
        else:
            self.indvidual_models = False
            self.model = args.Model
            self.ret_model = [args.Model]

        if args.Look_Ahead == 0:
            self.indvidual_look_ahead = True
            self.ret_look_ahead = [7, 14, 21]
            look_ahead_str = ''
        else:
            self.indvidual_look_ahead = False
            self.look_ahead = args.Look_Ahead
            self.ret_look_ahead = np.asarray([args.Look_Ahead])
            look_ahead_str = '_' + str(args.Look_Ahead) + 'LA'

        if not args.Server:
            self.logging_directory = '/Users/michael/Documents/github/Forecasting/Logging/'
        else:
            self.logging_directory = '/home/mimorris/Forecasting/Logging/'
        self.save_directory = self.logging_directory + args.Model + look_ahead_str + timestamp

        self.train_stats = pd.DataFrame(index=['MAE', 'RMSE', 'R', 'Lag'])
        self.test_predictions = pd.DataFrame()
        self.test_ground_truth = pd.DataFrame()
        self.stddev = pd.DataFrame()
        self.cleanup()

    def get_inputs(self):
        return self.ret_model, self.ret_look_ahead, self.ret_max_k

    def update_details(self, fold_num, model=None, look_ahead=None, k=None, epochs=None):
        self.fold_num = fold_num
        fold_str = str(2013 + fold_num) + '/' + str(14 + fold_num)

        if self.indvidual_models:
            model_str = str(model)
        else:
            model_str = ''

        if self.indvidual_look_ahead:
            look_ahead_str = '_' + str(look_ahead) + '_'
        else:
            look_ahead_str = ''

        if self.iter:
            iter_str = '_' + str(k)
        else:
            iter_str = ''

        if epochs:
            epochs_str = '_' + str(epochs)
        else:
            epochs_str = ''

        self.save_name = model_str + look_ahead_str + fold_str + iter_str + epochs_str

    def log(self, y_pred, y_true, model, stddev=None,save=False, save_weights=False, col_names=None):

        self.model_history = pd.DataFrame(model.history.history)
        if y_pred.ndim == 3:
            y_pred = np.squeeze(y_pred)
        if y_true.ndim == 3:
            y_true = np.squeeze(y_true)
        if y_true.ndim == 2:
            y_true = y_true[:, -1]
        if y_pred.ndim == 2:
            y_pred = y_pred[:, -1]

        if stddev is not None:
            self.do_stddev = True
            if stddev.ndim > 1:
                if stddev.shape[1] != 0:
                    stddev = stddev[:, -1]
                else:
                    stddev = np.squeeze(stddev)

            stddev = np.append(stddev, np.nan)[:366]
            self.stddev[str(self.save_name)] = stddev

        y_pred = np.asarray(y_pred)
        self.train_stats[str(self.save_name)] = metrics.evaluate(y_true, y_pred)

        y_true = np.append(y_true, np.nan)[:366]
        y_pred = np.append(y_pred, np.nan)[:366]

        self.test_ground_truth[str(self.save_name)] = y_true
        self.test_predictions[str(self.save_name)] = y_pred

        if save_weights:
            final_weights = model.weights[0].numpy()[-167:]
            weights = pd.DataFrame(columns=['weight'],index = np.asarray(col_names), data = np.squeeze(final_weights))
            weights.to_csv('weights.csv')

        if save:
            self.save(model)

    def cleanup(self):
        root = self.logging_directory
        folders = list(os.walk(root))[1:]

        for folder in folders:
            folder[1]
            # folder example: ('FOLDER/3', [], ['file'])
            if (len(folder[2]) == 0) and (len(folder[1]) == 0):
                os.rmdir(folder[0])

    def save(self, model):
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
        os.chdir(self.save_directory)

        if not os.path.exists(self.save_directory + '/models'):
            os.makedirs(self.save_directory + '/models')
        os.chdir(self.save_directory + '/models')

        self.model_history.to_csv(r'' + self.save_name.replace('/', '_') + '.csv')
        if self.save_model:
            model.save(self.save_name.replace('/', '_'), save_format='tf')
        os.chdir(self.save_directory)

        if self.do_stddev:
            self.stddev.to_csv(r'test_stddev.csv')

        self.train_stats.to_csv(r'train_stats.csv')
        self.test_predictions.to_csv(r'test_predictions.csv')
        self.test_ground_truth.to_csv(r'test_ground_truth.csv')
        os.chdir(self.root_directory)


