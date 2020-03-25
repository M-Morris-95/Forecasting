import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from make_data import *
tf.config.experimental_run_functions_eagerly(True)


def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  # c = np.log(np.expm1(1.))
  return tf.keras.Sequential([
      tfp.layers.VariableLayer(2 * n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(# pylint: disable=g-long-lambda
          tfd.Normal(loc=t[..., :n],
                     scale=1e-5 + tf.math.abs(t[..., n:])),
          reinterpreted_batch_ndims=1)),
  ])


def prior_trainable(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  return tf.keras.Sequential([
      tfp.layers.VariableLayer(n, dtype=dtype),
      tfp.layers.DistributionLambda(
          lambda t: tfd.Independent(tfd.Normal(loc=t, scale=1),# pylint: disable=g-long-lambda
                                    reinterpreted_batch_ndims=1)),
  ])

tfd = tfp.distributions

xlim=[0, 5]
ylim=[0, 1]
split = 0.25
length = 500
batch_size = 32
kl_weight = 1/batch_size
epochs = 500

# get data
x_train, y_train, x_test, y_test, y_stddev, y_mean = get_data(length=length, width=1, mean=0, stddev=0.4, split=split,
                                                      xlim=xlim, ylim=ylim)

# build model
negloglik = lambda y, p_y: -p_y.log_prob(y)

model = tf.keras.Sequential([
    tfp.layers.DenseVariational(1, posterior_mean_field, prior_trainable, kl_weight=kl_weight),
    tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=negloglik)

model.fit(x_train, y_train,
          epochs=epochs, batch_size=batch_size)

# predict train
yhat = np.asarray([model(x_train).mean() for i in range(100)])
mean = np.squeeze(np.mean(yhat, 0))
stddev = np.squeeze(np.std(yhat, 0))

# plot train
plt.scatter(x_train, y_train, alpha=0.5, color='orange')
plt.plot(x_train, mean, color='green', linewidth=2)
plt.plot(x_train, mean + stddev, color='green', linewidth=1)
plt.plot(x_train, mean - stddev, color='green', linewidth=1)

# predict test
yhat = np.asarray([model(x_test).mean() for i in range(100)])
mean = np.squeeze(np.mean(yhat, 0))
stddev = np.squeeze(np.std(yhat, 0))

# plot test
plt.scatter(x_test, y_test, alpha=0.5, color='orange')
plt.plot(x_test, mean, color='green', linewidth=2)
plt.plot(x_test, mean + stddev, color='green', linewidth=1)
plt.plot(x_test, mean - stddev, color='green', linewidth=1)

# tidy up graph
plt.plot([xlim[1]*(1-split), xlim[1]*(1-split)], [-100, 100], color = 'black')
plt.ylim((min(y_train), max(y_train)))
plt.annotate('train', (xlim[1]*(1-split)/2,max(y_train)*0.9))
plt.annotate('test', (xlim[1] - xlim[1]*split/2,max(y_train)*0.9))
plt.show()

