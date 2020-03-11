import pandas as pd
import numpy as np
import datetime


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