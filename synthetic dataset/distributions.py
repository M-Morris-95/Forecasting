import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions
tfk = tf.keras


# def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
#     n = kernel_size + bias_size
#
#     return tf.keras.Sequential([
#         tfp.layers.VariableLayer(2 * n, dtype=dtype),
#         tfp.layers.DistributionLambda(lambda t: tfd.Independent(
#             tfd.Normal(loc=t[..., :n],
#                        scale=1e-5 + tf.nn.softplus(0.5+t[..., n:])),
#             reinterpreted_batch_ndims=1))
#     ])
#
# def prior_trainable(kernel_size, bias_size=0, dtype=None):
#   n = kernel_size + bias_size
#   return tf.keras.Sequential([
#       tfp.layers.VariableLayer(n, dtype=dtype),
#       tfp.layers.DistributionLambda(
#           lambda t: tfd.Independent(tfd.Normal(loc=t, scale=1),
#                                     reinterpreted_batch_ndims=1)),
#   ])


def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  c = np.log(np.expm1(1.))
  return tf.keras.Sequential([
      tfp.layers.VariableLayer(2 * n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(# pylint: disable=g-long-lambda
          tfd.Normal(loc=t[..., :n],
                     scale=1e-5 + tf.nn.softplus(t[..., n:])),
          reinterpreted_batch_ndims=1)),
  ])


def prior_trainable(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  scale = 1
  return tf.keras.Sequential([
      tfp.layers.VariableLayer(n, dtype=dtype),
      tfp.layers.DistributionLambda(
          lambda t: tfd.Independent(tfd.Normal(loc=t, scale=scale),# pylint: disable=g-long-lambda
                                    reinterpreted_batch_ndims=1)),
  ])