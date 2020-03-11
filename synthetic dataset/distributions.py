import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras
import numpy as np

def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  c = np.log(np.expm1(1.)) #biases the softplus so that if t = 0 then scale = 1 + 1e-5
  return tf.keras.Sequential([
      tfp.layers.VariableLayer(2 * n, dtype=dtype, activation='linear'),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(  # pylint: disable=g-long-lambda
          tfd.Normal(loc=t[..., :n],
                     scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
          reinterpreted_batch_ndims=1)),
  ])


def prior_trainable(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  return tf.keras.Sequential([
      tfp.layers.VariableLayer(n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t, scale=1),
          reinterpreted_batch_ndims=1)),
  ])

def normal_scale_uncertainty(t, softplus_scale=0.2):
    """Create distribution with variable mean and variance"""
    return tfd.Normal(loc=t[..., :1],
                      scale=1e-3*tf.math.softplus(softplus_scale * t[..., 1:]))
