import numpy as np
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import tensorflow as tf
from make_data import *
from distributions import *
from Variational_Inference import *
from tensorflow.python import debug as tf_debug

# tf_debug.LocalCLIDebugWrapperSession(sess)t *
error = []
stddev_choices = np.linspace(0.1,1,19)

for stddev_in in stddev_choices:
    tfd = tfp.distributions
    xlim=[0, 1]
    ylim=[0, 1]
    split = 0.25
    length = 500
    iterations = 20000
    batch_size = 32
    kl_weight = 1/batch_size

    epochs = int(iterations / (length*(1-split)/batch_size))

    x_train, y_train, x_test, y_test, y_stddev, y_mean = get_data(length=length, width=1, mean=0, stddev=stddev_in, split=split,
                                                          xlim=xlim, ylim=ylim)

    negloglik = lambda y, p_y: -p_y.log_prob(y)

    model = tf.keras.Sequential([
        tfp.layers.DenseVariational(1, posterior_mean_field, prior_trainable, kl_weight=kl_weight),
        tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=negloglik)

    model.fit(x_train, y_train,
              epochs=epochs, batch_size=batch_size)

    lines = 100
    # Make predictions.
    yhat = [model(x_train).mean() for i in range(lines)]
    yhat = np.asarray(yhat)
    mean = np.squeeze(np.mean(yhat, 0))
    stddev = np.squeeze(np.std(yhat, 0))

    plt.scatter(x_train, y_train, alpha=0.5, color='orange')
    # for i in range(yhat.shape[0]):
    #     plt.plot(np.squeeze(x_train), yhat[i, :, :], linewidth=0.5, alpha=0.5, color='red')
    plt.plot(x_train, mean, color='green', linewidth=2)
    plt.plot(x_train, mean + stddev, color='green', linewidth=1)
    plt.plot(x_train, mean - stddev, color='green', linewidth=1)



    plt.plot([(1-split) * np.sum(xlim), (1-split) * np.sum(xlim)], [-1e3, 1e3], color='black')

    lines = 100
    # Make predictions.
    yhat = [model(x_test).mean() for i in range(lines)]
    yhat = np.asarray(yhat)
    mean = np.squeeze(np.mean(yhat, 0))
    stddev = np.squeeze(np.std(yhat, 0))

    plt.scatter(x_test, y_test, alpha=0.5, color='orange')
    # for i in range(yhat.shape[0]):
    #     plt.plot(np.squeeze(x_test), yhat[i, :, :], linewidth=0.5, alpha=0.5, color='red')
    plt.plot()
    plt.plot(x_test, mean, color='green', linewidth=2)
    plt.plot(x_test, mean + stddev, color='green', linewidth=1)
    plt.plot(x_test, mean - stddev, color='green', linewidth=1)

    plt.plot(np.concatenate([x_train, x_test]), y_mean, color='blue', linewidth=2)
    plt.plot(np.concatenate([x_train, x_test]), y_mean + y_stddev, color='blue', linewidth=1)
    plt.plot(np.concatenate([x_train, x_test]), y_mean - y_stddev, color='blue', linewidth=1)

    plt.ylim((np.min(y_train), np.max(y_test)))
    plt.show()

    weights_ideal = [ylim[1]/xlim[1], 0, y_stddev[-1], 0]
    weights_true = model.weights[0].numpy()
    error.append(np.mean(np.abs((weights_ideal - weights_true)/weights_true)))

