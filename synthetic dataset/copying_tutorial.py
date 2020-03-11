import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from distributions import *
tfk = tf.keras
tfd = tfp.distributions

def create_dataset():
  np.random.seed(43)
  n = 150
  w0 = 0.125
  b0 = 5.
  x_range = [-20, 60]
  x_tst = np.linspace(*x_range).astype(np.float32)

  def s(x):
    g = (x - x_range[0]) / (x_range[1] - x_range[0])
    return 3 * (0.25 + g**2.)

  x = (x_range[1] - x_range[0]) * np.random.rand(n) + x_range[0]
  eps = np.random.randn(n) * s(x)
  y = (w0 * x * (1. + np.sin(x)) + b0) + eps
  x = x[..., np.newaxis]
  x_tst = x_tst[..., np.newaxis]

  return y, x, x_tst


y, x, x_tst = create_dataset()

negloglik = lambda y, p_y: -p_y.log_prob(y)

case = 3

if case == 1:
    # Build model.
    model = tf.keras.Sequential([
      tf.keras.layers.Dense(1),
      tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
    ])

    # Do inference.
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05), loss=negloglik)
    model.fit(x, y, epochs=500, verbose=False)

    # Make predictions.
    y_pred = model(x_tst)

    plt.scatter(x, y)
    plt.plot(x_tst, y_pred.mean(), color='red')
    plt.ylim((0, 17.5))
    plt.show()

elif case == 2:
    # Build model.
    model = tfk.Sequential([
        tf.keras.layers.Dense(1 + 1),
        tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t[..., :1],
                                 scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]))),
    ])

    # Do inference.
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05), loss=negloglik)
    model.fit(x, y, epochs=500, verbose=False, batch_size=8)

    # Make predictions.
    yhat = model(x_tst)

    mean = np.squeeze(yhat.mean().numpy())
    stddev = np.squeeze(yhat.stddev().numpy())

    plt.scatter(x, y)
    plt.plot(x_tst, yhat.mean(), color='red')
    plt.fill_between(np.squeeze(x_tst), mean-2*stddev,  mean+2*stddev,
                     color="pink", alpha=0.5, label="train predict std")
    plt.ylim((0, 17.5))
    plt.show()

elif case == 3:
    # Build model.
    model = tf.keras.Sequential([
        tfp.layers.DenseVariational(1, posterior_mean_field, prior_trainable),
        tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
    ])

    # Do inference.
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05), loss=negloglik)
    model.fit(x, y, epochs=500, verbose=False, batch_size=32)

    # Make predictions.
    yhat = [model(x_tst).mean() for i in range(100)]
    yhat = np.asarray(yhat)

    mean = np.squeeze(np.mean(yhat,0))
    stddev = np.squeeze(np.std(yhat,0))

    plt.scatter(x, y)
    for i in range(yhat.shape[0]):
        plt.plot(np.squeeze(x_tst), yhat[i,:,:], linewidth=0.5, alpha=0.5,  color='red')
    plt.plot(x_tst, mean, color='red')
    plt.ylim((0, 17.5))
    plt.show()
