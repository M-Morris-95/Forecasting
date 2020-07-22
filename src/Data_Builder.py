import pandas as pd
import numpy as np
import datetime

class data_builder:
    def __init__(self, args, fold, out_of_sample=False):
        self.fold=fold
        self.out_of_sample = out_of_sample
        self.args = args

        if not args.Server:
            self.weather_directory = '/Users/michael/Documents/ili_data/Weather/all_weather_data.csv'
            self.directory = '/Users/michael/Documents/ili_data/dataset_forecasting_lag' + str(
                args.Lag) + '/' + args.Country + '_smoothed_' + str(args.Look_Ahead) + '/fold' + str(fold) + '/'
        else:
            self.weather_directory = '/home/mimorris/ili_data/Weather/all_weather_data.csv'
            self.directory = '/home/mimorris/ili_data/dataset_forecasting_lag' + str(
                args.Lag) + '/' + args.Country + '_smoothed_' + str(args.Look_Ahead) + '/fold' + str(fold) + '/'

    def load_ili_data(self, path):
        ili_data = pd.read_csv(path, header=None, names=['ili'], index_col=0, parse_dates=True)
        return ili_data

    def load_google_data(self, path):
        weather_data = pd.read_csv(self.weather_directory, index_col=0, parse_dates=True)
        google_data = pd.read_csv(path, index_col=0, parse_dates=True)
        if self.args.Weather: google_data = google_data.join(weather_data)
        if self.args.DOTY: google_data['doty'] = np.asarray([val.timetuple().tm_yday for val in google_data.index])
        return google_data

    def window(self, data):
        windowed = []
        for i in range(1+data.shape[0] - self.args.Lag):
            windowed.append(data.iloc[i:i + self.args.Lag].values)
        windowed = np.asarray(windowed)
        return windowed

    def mimo(self, y, ground_truth):

        data = []
        if self.args.MIMO:
            for i in range(y.shape[0]):
                start = y.index[i] - datetime.timedelta(self.args.MIMO)
                end = y.index[i] + datetime.timedelta(self.args.MIMO)
                data.append(ground_truth.iloc[np.argwhere(ground_truth.index == start)[0][0]:np.argwhere(ground_truth.index == end)[0][0]].values.squeeze())

            data = np.asarray(data)
            columns = []
            for i in range(-self.args.MIMO, self.args.MIMO, 1):
                if i > 0:
                    add = '+'
                else:
                    add = ''
                columns.append('T' + add + str(i))
            data = pd.DataFrame(data=data, columns=columns, index=y.index)
            return data
        else:
            return y

    def build(self, normalise_all=False):
        google_train = self.load_google_data(self.directory + 'google-train')
        google_train['ili'] = self.load_ili_data(self.directory + 'ili-train')['ili'].values
        y_train = self.load_ili_data(self.directory + 'y-train')

        google_test = self.load_google_data(self.directory + 'google-test')
        google_test['ili'] = self.load_ili_data(self.directory + 'ili-test')['ili'].values
        y_test = self.load_ili_data(self.directory + 'y-test')

        ili_ground_truth = pd.read_csv(
            '/Users/michael/Documents/ili_data/ili_ground_truth/ILI_rates_UK_thursday_linear_interpolation_new.csv',
            parse_dates=True, index_col=['date'])

        # google_oos = self.load_google_data('/Users/michael/Documents/datasets/covid flu tweets/google-test_fold'+str(self.fold))
        # google_oos['ili'] = self.load_ili_data('/Users/michael/Documents/datasets/covid flu tweets/ili-test_fold'+str(self.fold))['ili'].values
        # y_oos = self.load_ili_data('/Users/michael/Documents/datasets/covid flu tweets/y-test_fold' + str(self.fold))

        if self.args.Square_Inputs:
            google_train = pd.concat((google_train, google_train.pow(2)), axis=1)
            google_test = pd.concat((google_test, google_test.pow(2)), axis=1)
            # google_oos = pd.concat((google_oos, google_oos.pow(2)), axis=1)

        n = normalizer(google_train, y_train, normalise_all)

        google_train = n.normalize(google_train)
        google_test = n.normalize(google_test)
        # google_oos = n.normalize(google_oos)

        x_train = self.window(google_train)
        x_test = self.window(google_test)
        # x_oos = self.window(google_oos)

        y_test = self.mimo(y_test, ili_ground_truth)
        y_train = self.mimo(y_train, ili_ground_truth)
        # y_oos = self.mimo(y_oos, ili_ground_truth)

        assert (x_train.shape[0] == y_train.shape[0])
        assert (x_test.shape[0] == y_test.shape[0])

        return x_train, y_train, x_test, y_test

class normalizer:
    def __init__(self, x, y, normalize_all=False):
        if normalize_all:
            self.x_min = np.min(np.asarray(x), axis=0)
            self.x_max = np.max(np.asarray(x), axis=0)

        else:
            self.x_min = np.min(np.asarray(x.iloc[:, :-1]), axis=0)
            self.x_max = np.max(np.asarray(x.iloc[:, :-1]), axis=0)

        self.normalize_all = normalize_all
        self.y_min = np.min(y)[0]
        self.y_max = np.max(y)[0]

    def normalize(self, X):
        if not self.normalize_all:
            x_val = np.asarray(X.iloc[:, :-1])
        else:
            x_val = np.asarray(X)

        for i in range(x_val.shape[0]):
            x_val[i] = (x_val[i] - self.x_min) / (self.x_max - self.x_min)

        if not self.normalize_all:
            x_val = np.concatenate([x_val, X.iloc[:, -1].values[:, np.newaxis]], 1)
        X_norm = pd.DataFrame(data=x_val, index = X.index, columns=X.columns)
        return X_norm

    def un_normalize(self, Y):
        y_val = np.asarray(Y[1])
        for i in range(y_val.shape[0]):
            y_val[i] = y_val[i] * (self.y_max - self.y_min) - self.y_min

        Y[1] = y_val
        return Y