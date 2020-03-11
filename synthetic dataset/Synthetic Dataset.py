import numpy as np
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import tensorflow as tf
from Plotter import *
from metrics import *
from make_data import *
from distributions import *
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

fig = plotter(1)

EPOCHS = 1000
length = 1028

x_train, y_train, x_test, y_test = get_data(length = length, stddev = 0.2, width = 1)
x_tst = x_train
# y_train, x_train, x_tst = create_dataset()
# (150, 1)
# num_batches = length / BATCH_SIZE

# kl_weight = 1.0 / num_batches
kl_weight = 1.0

negloglik = lambda y, p_y: -p_y.log_prob(y)


model = tf.keras.Sequential([
    tfp.layers.DenseVariational(1, posterior_mean_field, posterior_mean_field),
    tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=negloglik)

model.fit(x_train, y_train,
                epochs=1500, batch_size=32)

# Make predictions.
yhat = [model(x_tst).mean() for i in range(100)]
yhat = np.asarray(yhat)

mean = np.squeeze(np.mean(yhat, 0))
stddev = np.squeeze(np.std(yhat, 0))

plt.scatter(x_train, y_train)
for i in range(yhat.shape[0]):
    plt.plot(np.squeeze(x_tst), yhat[i, :, :], linewidth=0.5, alpha=0.5, color='red')
plt.plot(x_tst, mean, color='red')
plt.ylim((0, 17.5))
plt.show()

fig.plot_synthetic(x_train, y_train, model, tfd = True)





# ili_input = tf.keras.layers.Input(shape=[x_train.shape[1], x_train.shape[2]])
# flatten = tf.keras.layers.Flatten()(ili_input)
#
# DenseVariational = tfp.layers.DenseVariational(1, make_posterior_fn=posterior_mean_field, make_prior_fn=prior_trainable)(flatten)
# DistributionLambda = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1))(DenseVariational)
#
# model = tf.keras.models.Model(inputs=ili_input, outputs=DistributionLambda)