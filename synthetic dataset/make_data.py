import numpy as np

def get_data(length=500, width=1, mean=0, stddev=1e-5, split=0.25, xlim=[0, 60], ylim=[0, 60]):
    x_train = np.zeros((length, width))
    y_train = np.zeros((length, 1))
    y_stddev = []
    y_mean = []
    for i in range(x_train.shape[0]):
        for j in range(x_train.shape[1]):
            x_train[i, j] = i
        temp = (x_train[i, 0] / length)
        y_train[i] = temp + (np.random.normal(loc = mean, scale = stddev * temp))
        y_mean.append(temp)
        y_stddev.append(stddev * temp)

    y_stddev = np.asarray(y_stddev)
    y_mean = np.asarray(y_mean)

    x_train = x_train / i * xlim[1] + xlim[0]
    y_train = y_train*ylim[1]+ylim[0]
    y_stddev = y_stddev*ylim[1]
    y_mean = y_mean *ylim[1]+ylim[0]

    split_te = int((split) * (length))

    return x_train[:-split_te], np.squeeze(y_train[:-split_te]), x_train[-split_te:], np.squeeze(
        y_train[-split_te:]), y_stddev, y_mean
