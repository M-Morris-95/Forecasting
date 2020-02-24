import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Parser import GetParser
from Functions import data_builder,logger, plotter


def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  c = np.log(np.expm1(1.0))

  return tf.keras.Sequential([
    tfp.layers.VariableLayer(2 * n, dtype=dtype),
    tfp.layers.DistributionLambda(lambda t: tfd.Independent(
      tfd.Normal(loc=t[..., :n],
                 scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
      reinterpreted_batch_ndims=1))
  ])


def prior_trainable(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size

  return tf.keras.Sequential([
    tfp.layers.VariableLayer(n, dtype=dtype),
    tfp.layers.DistributionLambda(lambda t: tfd.Independent(
      tfd.Normal(loc=t, scale=1.0),
      reinterpreted_batch_ndims=1))
  ])
plt.figure(num=None, figsize=(10, 8), dpi=200, facecolor='w', edgecolor='k')
for foldnum in range(1,5):
  optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)
  parser = GetParser()
  args = parser.parse_args()

  tfd = tfp.distributions
  negloglik = lambda y, p_y: -p_y.log_prob(y)

  data = data_builder(args, fold=foldnum, look_ahead=14)
  x_train, y_train, y_train_index, x_test, y_test, y_test_index = data.build()
  x_train = x_train.reshape((x_train.shape[0], -1))
  x_test = x_test.reshape((x_test.shape[0], -1))
  y_test = y_test[:,-1]
  y_train = y_train[:,-1]


  ili_input = tf.keras.layers.Input(shape=[x_train.shape[1]])
  output = tf.keras.layers.Dense(1)(ili_input)
  linear_model = tf.keras.models.Model(inputs=ili_input, outputs=output)

  linear_model.compile(optimizer=optimizer,
                loss='mae',
                metrics=['mae', 'mse'])

  model1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
    tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
  ])
  model1.compile(optimizer=optimizer, loss=negloglik)


  model2 = tf.keras.Sequential([
    tf.keras.layers.Dense(1 + 1),
    tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1],
                             scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]))),
  ])
  model2.compile(optimizer=optimizer, loss=negloglik)

  # Build model.
  model3 = tf.keras.Sequential([
    tfp.layers.DenseVariational(1,
                                posterior_mean_field,
                                prior_trainable),
    tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
  ])
  model3.compile(optimizer=optimizer, loss=negloglik)


  model4 = tf.keras.Sequential([
      tfp.layers.DenseVariational(2,
                                  posterior_mean_field,
                                  prior_trainable),
      tfp.layers.DistributionLambda(
                                  lambda t: tfd.Normal(loc=t[..., :1],
                                  scale=1e-3 + tf.math.softplus(0.01 * t[..., 1:])))
      ])

  model4.compile(optimizer=optimizer,
                       loss=negloglik)



  linear_model.fit(x_train, y_train,
             epochs=100, batch_size=128,
             verbose=2)

  # model1.fit(x_train, y_train,
  #            epochs=100, batch_size=128,
  #            verbose=2)

  model2.fit(x_train, y_train,
             epochs=100, batch_size=128,
             verbose=2)

  # model3.fit(x_train, y_train,
  #            epochs=100, batch_size=128,
  #            verbose=2)
  #
  # model4.fit(x_train, y_train,
  #            epochs=100, batch_size=128,
  #            verbose=2)

  # yhats = [model(x_tst) for i in range(100)]

  fig.plot(foldnum, y1, y2, x1=False):

  plt.subplot(2,2,foldnum)
  yhat = model2(x_test)
  mean = yhat.mean()
  stddev = yhat.stddev()
  mean_plus_2_stddev = mean - stddev
  mean_minus_2_stddev = mean + stddev
  # np.linspace(1, x_test.shape[0], x_test.shape[0])
  mean = yhat.mean()
  X = pd.to_datetime(y_test_index)

  plt.plot(mean, color='red', label='Gaussian')
  plt.plot(linear_model(x_test), color='green', label='Linear')
  plt.fill_between(np.linspace(1, x_test.shape[0], x_test.shape[0]), np.squeeze(mean_minus_2_stddev), np.squeeze(mean_plus_2_stddev),
                   color="pink", alpha=0.5, label="Confidence Interval")
  plt.xlabel('Day of the Season', fontsize=8)
  plt.ylabel('ILI Rate (Infected/100,000)', fontsize=8)
  plt.plot(y_test, label="Ground Truth")
  plt.legend(fontsize=8)
plt.show()