import tensorflow as tf
from scipy.stats import pearsonr
import numpy as np

def pearson(y_true, y_pred):
    if type(y_pred) != np.ndarray:
        y_pred = y_pred.numpy()
    y_pred = y_pred.astype('float64')

    # y_true = y_true.squeeze()
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

def evaluate(y_true, y_pred):
    return mae(y_true, y_pred).numpy(), rmse(y_true, y_pred).numpy(), pearson(y_true, y_pred)