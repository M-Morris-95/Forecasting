from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dropout, Conv1D, GRU, Attention, Dense, Input, concatenate, Flatten, Layer, \
    LayerNormalization, Embedding, Dropout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import metrics
from Transformer import MultiHeadAttention
import datetime
import os
import time
import tensorflow as tf

def noised_inputs(model, x, num=20, stddev = 0.1):
    prediction = []
    for i in range(num):
        prediction.append(model.predict(x + np.random.normal(0, stddev, x.shape)))

    prediction = np.squeeze(np.asarray(prediction))

    mean = np.mean(prediction, axis = 0)
    std = np.std(prediction, axis = 0)

    return mean, std


def full_GRU(x_train, y_train):
    initializer = tf.keras.initializers.glorot_normal(seed=None)
    ili_input = Input(shape=[x_train.shape[1], 1])

    x = GRU(28, activation='relu', return_sequences=True, kernel_initializer=initializer)(ili_input)
    x = Model(inputs=ili_input, outputs=x)

    google_input = Input(shape=[x_train.shape[1], x_train.shape[2] - 1])
    y = GRU(x_train.shape[2] - 1, activation='relu', return_sequences=True, kernel_initializer=initializer)(
        google_input)
    y = GRU(int(0.5 * (x_train.shape[2] - 1)), activation='relu', return_sequences=True,
            kernel_initializer=initializer)(y)
    y = Model(inputs=google_input, outputs=y)

    z = concatenate([x.output, y.output])
    z = GRU(y_train.shape[1], activation='relu', return_sequences=False, kernel_initializer=initializer)(z)

    model = Model(inputs=[x.input, y.input], outputs=z)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)

    model.compile(optimizer=optimizer,
                  loss='mae',
                  metrics=['mae', 'mse', metrics.rmse])

    return model


def simple(x_train):
    ili_input = Input(shape=[x_train.shape[1], x_train.shape[2]])
    flatten = tf.keras.layers.Flatten()(ili_input)
    output = tf.keras.layers.Dense(1)(flatten)
    model = Model(inputs=ili_input, outputs=output)

    return model


def recurrent_attention(x_train, y_train, num_heads=1, regularizer=False):
    if regularizer:
        regularizer = tf.keras.regularizers.l2(0.01)
    else:
        regularizer = None

    d_model = x_train.shape[1]

    ili_input = Input(shape=[x_train.shape[1], x_train.shape[2]])
    x = GRU(x_train.shape[1], activation='relu', return_sequences=True, kernel_regularizer=regularizer)(ili_input)

    x = MultiHeadAttention(d_model, num_heads, name="attention", regularizer=regularizer)({
        'query': x,
        'key': x,
        'value': x
    })
    x = GRU(int((x_train.shape[2] - 1)), activation='relu', return_sequences=True, kernel_regularizer=regularizer)(x)
    y = GRU(int(0.75 * (x_train.shape[2] - 1)), activation='relu', return_sequences=False,
            kernel_regularizer=regularizer)(x)
    y = tf.keras.layers.RepeatVector((y_train.shape[1]))(y)
    z = GRU(1, activation='relu', return_sequences=True, kernel_regularizer=regularizer)(y)
    model = Model(inputs=ili_input, outputs=z)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)

    model.compile(optimizer=optimizer,
                  loss='mae',
                  metrics=['mae', 'mse', metrics.rmse])

    return model


def build_attention(x_train, y_train, num_heads=1, regularizer=False, initializer=None):
    if regularizer:
        regularizer = tf.keras.regularizers.l2(0.01)
    else:
        regularizer = None

    d_model = x_train.shape[1]

    ili_input = Input(shape=[x_train.shape[1], x_train.shape[2]])
    x = GRU(x_train.shape[1], activation='relu', return_sequences=True, kernel_regularizer=regularizer, kernel_initializer=initializer)(ili_input)

    x = MultiHeadAttention(d_model, num_heads, name="attention", regularizer=regularizer, initializer=initializer)({
        'query': x,
        'key': x,
        'value': x
    })
    x = GRU(int((x_train.shape[2] - 1)), activation='relu', return_sequences=True, kernel_regularizer=regularizer, kernel_initializer=initializer)(x)
    y = GRU(int(0.75 * (x_train.shape[2] - 1)), activation='relu', return_sequences=True,
            kernel_regularizer=regularizer, kernel_initializer=initializer)(x)
    z = GRU(y_train.shape[1], activation='relu', return_sequences=False, kernel_regularizer=regularizer, kernel_initializer=initializer)(y)

    model = Model(inputs=ili_input, outputs=z)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)

    model.compile(optimizer=optimizer,
                  loss='mae',
                  metrics=['mae', 'mse', metrics.rmse])

    return model


def simple_GRU(x_train, y_train, regularizer=False):
    if regularizer:
        regularizer = tf.keras.regularizers.l2(0.01)
    else:
        regularizer = None

    batch_norm = True

    ili_input = Input(shape=[x_train.shape[1], x_train.shape[2]])
    x = GRU(x_train.shape[1], activation='relu', return_sequences=True, kernel_regularizer=regularizer)(ili_input)
    if batch_norm: x = tf.keras.layers.BatchNormalization()(x)
    x = GRU(int((x_train.shape[2] - 1)), activation='relu', return_sequences=True, kernel_regularizer=regularizer)(x)
    if batch_norm: x = tf.keras.layers.BatchNormalization()(x)
    x = GRU(int(0.75 * (x_train.shape[2] - 1)), activation='relu', return_sequences=True,
            kernel_regularizer=regularizer)(x)
    if batch_norm: x = tf.keras.layers.BatchNormalization()(x)
    z = GRU(y_train.shape[1], activation='linear', return_sequences=False, kernel_regularizer=regularizer)(x)

    model = Model(inputs=ili_input, outputs=z)

    return model
