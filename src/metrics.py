import tensorflow as tf
from scipy.stats import pearsonr
from scipy import special
import numpy as np

def pearson(y_true, y_pred):
    if type(y_pred) != np.ndarray:
        y_pred = y_pred.numpy()
    y_pred = y_pred.astype('float64')

    y_true = y_true.astype('float64')
    corr = pearsonr(y_true, y_pred)[0]
    return corr

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true-y_pred)))

def rse(y_true, y_pred):
    num = tf.sqrt(tf.reduce_mean(tf.square(y_true-y_pred)))
    den = tf.math.reduce_std(y_true)
    return num/den

def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true-y_pred))

def mae(y_true, y_pred):
    return tf.reduce_mean(tf.math.abs(y_true-y_pred))

def lag(y_true, y_pred):
    return tf.argmax(y_pred)-tf.argmax(y_true)

_normconst = 1.0 / np.sqrt(2.0 * np.pi)
# Cumulative distribution function of a univariate standard Gaussian
# distribution with zero mean and unit variance.
_normcdf = special.ndtr
def _normpdf(x):
    """Probability density function of a univariate standard Gaussian
    distribution with zero mean and unit variance.
    """
    return _normconst * np.exp(-(x * x) / 2.0)

def crps_gaussian(x, mu, sig, grad=False):
    """
    Computes the CRPS of observations x relative to normally distributed
    forecasts with mean, mu, and standard deviation, sig.
    CRPS(N(mu, sig^2); x)
    Formula taken from Equation (5):
    Calibrated Probablistic Forecasting Using Ensemble Model Output
    Statistics and Minimum CRPS Estimation. Gneiting, Raftery,
    Westveld, Goldman. Monthly Weather Review 2004
    http://journals.ametsoc.org/doi/pdf/10.1175/MWR2904.1
    Parameters
    ----------
    x : scalar or np.ndarray
        The observation or set of observations.
    mu : scalar or np.ndarray
        The mean of the forecast normal distribution
    sig : scalar or np.ndarray
        The standard deviation of the forecast distribution
    grad : boolean
        If True the gradient of the CRPS w.r.t. mu and sig
        is returned along with the CRPS.
    Returns
    -------
    crps : scalar or np.ndarray or tuple of
        The CRPS of each observation x relative to mu and sig.
        The shape of the output array is determined by numpy
        broadcasting rules.
    crps_grad : np.ndarray (optional)
        If grad=True the gradient of the crps is returned as
        a numpy array [grad_wrt_mu, grad_wrt_sig].  The
        same broadcasting rules apply.
    """
    x = np.asarray(x)
    mu = np.asarray(mu)
    sig = np.asarray(sig)
    # standadized x
    sx = (x - mu) / sig
    # some precomputations to speed up the gradient
    pdf = _normpdf(sx)
    cdf = _normcdf(sx)
    pi_inv = 1. / np.sqrt(np.pi)
    # the actual crps
    crps = sig * (sx * (2 * cdf - 1) + 2 * pdf - pi_inv)
    if grad:
        dmu = 1 - 2 * cdf
        dsig = 2 * pdf - pi_inv
        return crps, np.array([dmu, dsig])
    else:
        return crps


def evaluate(y_true, y_pred):
    return mae(y_true, y_pred).numpy(), rmse(y_true, y_pred).numpy(), pearson(y_true, y_pred), lag(y_true, y_pred)