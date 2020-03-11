import numpy as np

def get_data(length = 500, width = 1, mean = 0, stddev = 0.1, split = 0.25):
    x_train = np.zeros((length,width))
    y_train = np.zeros((length, 1))

    xmax = 60
    xmin = -20

    for j in range(x_train.shape[1]):
        for i in range(x_train.shape[0]):
            x_train[i, j] = i +(np.random.normal(mean, 0.1))

    for i in range(y_train.shape[0]):
        temp = (x_train[i, 0] / length)
        y_train[i] = temp + 0.5*(np.random.normal(mean, stddev))

    x_train = x_train/(np.max(x_train)/xmax) + xmin
    y_train = y_train*20


    split_te = int((split)*(length))

    x_test = x_train[-split_te:]
    y_test = y_train[-split_te:]

    x_train= x_train[:-split_te]
    y_train = y_train[:-split_te]

    return x_train, np.squeeze(y_train), x_test, y_test