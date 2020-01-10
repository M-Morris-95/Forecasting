import pandas as pd
import os
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import argparse

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dropout, Conv1D, GRU, Attention, Dense, Input, concatenate, Flatten, Layer, LayerNormalization, Embedding, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from scipy.stats import pearsonr

class normalizer:
    def __init__(self, x_train, y_train):
        self.x_min = np.min(np.asarray(x_train), axis=0)
        self.x_max = np.max(np.asarray(x_train), axis=0)

        self.y_min = np.min(np.asarray(y_train), axis=0)[1]
        self.y_max = np.max(np.asarray(y_train), axis=0)[1]

    def normalize(self, X, Y):
        x_val=np.asarray(X)
        for i in range(x_val.shape[0]):
            x_val[i] = (x_val[i]-self.x_min)/(self.x_max - self.x_min)
        X_norm = pd.DataFrame(data=x_val, columns=X.columns)

        return X_norm

    def un_normalize(self, Y):
        y_val = np.asarray(Y[1])
        for i in range(y_val.shape[0]):
            y_val[i] = y_val[i] * (self.y_max - self.y_min) - self.y_min

        Y[1] = y_val
        return Y

class plotter:
    def __init__(self, number):
        self.number = number
        plt.figure(number, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')

    def plot(self, fold_num, y1, y2, x1 = False):
        plt.figure(self.number)
        plt.subplot(2, 2, fold_num)
        if type(x1) != np.ndarray:
            plt.plot(y1)
            plt.plot(y2)
        else:
            plt.plot(pd.to_datetime(x1), y1)
            plt.plot(pd.to_datetime(x1), y2)
            plt.legend(['prediction', 'true'])

        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    def save(self, name):

        plt.figure(self.number)
        plt.savefig(name)

    def show(self):
        plt.figure(self.number)
        plt.show()

def GetParser():
    parser = argparse.ArgumentParser(
        description='M-Morris-95 Foprecasting')

    parser.add_argument('--Server', '-S',
                        type=bool,
                        help='is it on the server?',
                        default=False,
                        required = False)

    return parser

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

def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true-y_pred))

def mae(y_true, y_pred):
    return tf.reduce_mean(tf.math.abs(y_true-y_pred))

def evaluate(y_true, y_pred):
    return mae(y_true, y_pred).numpy(), rmse(y_true, y_pred).numpy(), pearson(y_true, y_pred)

def load_google_data(path):
    google_data = pd.read_csv(path)
    google_data = google_data.drop(['Unnamed: 0'], axis=1)
    return google_data

def load_ili_data(path):
    ili_data = pd.read_csv(path, header=None)
    ili_data = ili_data[1]
    return ili_data

def build_data(fold_dir):
    google_train = load_google_data(fold_dir + 'google-train')
    google_train['ili'] = load_ili_data(fold_dir + 'ili-train').values

    google_test = load_google_data(fold_dir + 'google-test')
    google_test['ili'] = load_ili_data(fold_dir + 'ili-test').values

    # y_train = pd.read_csv(fold_dir + 'y-train', header=None)
    # y_train = np.asarray(y_train[1])
    # y_test = pd.read_csv(fold_dir + 'y-test', header=None)
    # y_test = np.asarray(y_test[1])

    temp = pd.read_csv(fold_dir + 'ili-train', header=None)[27:48]
    temp = temp.append(pd.read_csv(fold_dir + 'y-train', header=None))
    temp = temp[1].values # first of y train is at index 20

    y_train = []
    for i in range(len(temp) - 21):
        y_train.append(np.asarray(temp[i:i + 21]))
    y_train = np.asarray(y_train)

    temp = pd.read_csv(fold_dir + 'ili-test', header=None)[27:48]
    temp = temp.append(pd.read_csv(fold_dir + 'y-test', header=None))
    temp = temp[1].values # first of y test is at index 20

    y_test = []
    for i in range(len(temp) - 21):
        y_test.append(np.asarray(temp[i:i + 21]))
    y_test = np.asarray(y_test)



    n = normalizer(google_train, y_train)
    google_train = n.normalize(google_train, y_train)
    google_test = n.normalize(google_test, y_test)

    x_train = []
    x_test = []
    lag = 28

    for i in range(len(google_train) - lag + 1):
        x_train.append(np.asarray(google_train[i:i + lag]))

    for i in range(len(google_test) - lag + 1):
        x_test.append(np.asarray(google_test[i:i + lag]))

    y_train_index = pd.read_csv(fold_dir + 'y-train', header=None)[0][:len(x_train)]
    x_train = np.asarray(x_train)
    y_train = y_train[:x_train.shape[0]]

    y_test_index = pd.read_csv(fold_dir + 'y-test', header=None)[0][:len(x_test)]
    x_test = np.asarray(x_test)
    y_test = y_test[:x_test.shape[0]]

    return x_train, y_train, y_train_index, x_test, y_test, y_test_index

def build_model(x_train):
    ili_input = Input(shape=[x_train.shape[1],1])
    x = GRU(28, activation='relu', return_sequences=True)(ili_input)
    x = Model(inputs=ili_input, outputs=x)

    google_input = Input(shape=[x_train.shape[1], x_train.shape[2]-1])
    y = GRU(x_train.shape[2]-1, activation='relu', return_sequences=True)(google_input)
    y = GRU(int(0.5*(x_train.shape[2]-1)), activation='relu', return_sequences=True)(y)
    y = Model(inputs=google_input, outputs=y)

    z = concatenate([x.output, y.output])

    #  I think that the output needs to be recurrent, size [None, 21, 1] but I don't know how to do that. Setting the
    #  output to that generates an error because the input is a time series with different dimensions. The goal is then
    #  that the previous output (t-1) is put back into an attention mechanism with concatenation. The attention works
    #  on the concatenated values which then go back into the output layer to give the prediction at time t.

    z = GRU(21, activation='relu',return_sequences=False)(z)


    model = Model(inputs=[x.input, y.input], outputs=z)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)

    model.compile(optimizer=optimizer,
                  loss='mae',
                  metrics=['mae', 'mse', rmse])

    return model

def scaled_dot_product_attention(query, key, value, mask=None):
    # Q K MatmMul
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # Q K Scale by sqrt dk
    dk = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(dk)

    # Do mask if included. add the mask zero out padding tokens.
    if mask is not None:
        logits += (mask * -1e9)

    # softmax Q K
    attention_weights = tf.nn.softmax(logits, axis=-1)

    # matmul Q K with V
    Attention = tf.matmul(attention_weights, value)

    return Attention

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model, activation='linear')#, name = 'query_dense')
        self.key_dense = tf.keras.layers.Dense(units=d_model, activation='linear')#, name = 'key_dense')
        self.value_dense = tf.keras.layers.Dense(units=d_model, activation='linear')#, name = 'value_dense')
        self.output_dense = tf.keras.layers.Dense(units=d_model, activation='linear')

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def __call__(self, inputs):
        query, key, value = inputs['query'], inputs['key'], inputs['value']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot product attention
        scaled_attention = scaled_dot_product_attention(query, key, value)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # Concat
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # Linear
        outputs = self.output_dense(concat_attention)
        return outputs

def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(units, d_model), name="inputs")

    # multi head attention
    attention = MultiHeadAttention(d_model, num_heads, name="attention")({
        'query': inputs,
        'key': inputs,
        'value': inputs
    })
    # dropout
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)

    # add and norm
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

    # feed forward
    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    # dropout
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)

    # add and norm
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

    # assemble layer
    layer = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    return layer

def encoder(num_layers, units, d_model, num_heads, dropout, name="encoder"):
    # create input
    inputs = tf.keras.Input(shape=(units,d_model), name="inputs")

    # no embeddings but if there were put them here.

    # add dropout
    outputs = tf.keras.layers.Dropout(rate=dropout)(inputs)

    # create layers
    for i in range(num_layers):
        outputs = encoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="encoder_layer_{}".format(i),
        )(outputs)

    # assemble model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    return model

def encoder_network(output_size, num_layers, units, d_model, num_heads, dropout, name="transformer"):
    # inputs
    inputs = tf.keras.Input(shape=(units,d_model), name="inputs")

    # encoder
    enc_outputs = encoder(
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[inputs])

    outputs = tf.keras.layers.Dense(units=output_size, activation = 'relu')(enc_outputs)
    # output dense layer
    outputs = Flatten()(outputs)
    outputs = tf.keras.layers.Dense(units=output_size, name="outputs")(outputs)
    # build model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    return model



parser = GetParser()
args = parser.parse_args()

timestamp = time.strftime('%b-%d-%Y-%H-%M', time.localtime())

fig1 = plotter(1)
fig3 = plotter(3)

results = pd.DataFrame(index = ['MAE', 'RMSE', 'R'])
test_predictions = pd.DataFrame()

if not args.Server:
    logging_dir = '/Users/michael/Documents/github/Forecasting/Logging/'
    data_dir = '/Users/michael/Documents/ili_data/dataset_forecasting_lag28/eng_smoothed_14/fold'
else:
    logging_dir = '/home/mimorris/Forecasting/Logging/'
    data_dir = '/home/mimorris/ili_data/dataset_forecasting_lag28/eng_smoothed_14/fold'

num_heads = [1,8,4,9,11]


for fold_num in range(1,5):
    fold_dir = data_dir + str(fold_num) + '/'
    save_dir = logging_dir + timestamp + '/Fold_' + str(fold_num)

    x_train, y_train, y_train_index, x_test, y_test, y_test_index  = build_data(fold_dir)

    model = build_model(x_train)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)

    model.compile(optimizer=optimizer,
                  loss='mse',
                  # loss = loss,
                  metrics=['mae', 'mse', rmse])

    earlystop_callback = EarlyStopping(
        monitor='val_loss', min_delta=0.0001,
        patience=5)

    model.fit(
        [x_train[:,:,-1, np.newaxis],x_train[:,:,:-1]], y_train,
        callbacks=[earlystop_callback],
        validation_data=([x_test[:,:,-1, np.newaxis],x_test[:,:,:-1]], y_test),
        epochs=100, batch_size=64)

    prediction = model.predict([x_test[:,:,-1, np.newaxis], x_test[:,:,:-1]])[:,20]
    y_test = y_test[:, -1]

    results[str(2014) + '/' + str(14 + fold_num)] = evaluate(y_test, prediction)
    test_predictions['prediction_'+str(2014) + '/' + str(14 + fold_num)] = prediction[:365]
    test_predictions['truth_' + str(2014) + '/' + str(14 + fold_num)] = y_test[:365]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    os.chdir(save_dir)

    model.save_weights('transformer.hdf5')

    training_stats = pd.DataFrame(model.history.history)
    training_stats.to_csv(r'Fold_'+str(fold_num)+'_training_stats.csv')

    fig1.plot(fold_num, training_stats.mae, training_stats.val_mae)
    fig3.plot(fold_num, prediction, y_test, y_test_index)




os.chdir(logging_dir + timestamp)
results.to_csv(r'stats.csv')
test_predictions.to_csv(r'test_predictions.csv')

fig1.save('training_stats.png')
fig3.save('validation_predictions.png')

fig1.show()
fig3.show()
