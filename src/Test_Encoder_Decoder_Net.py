import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from Parser import GetParser
from Data_Builder import *
from Plotter import *
from Logger import *
from Models import *

parser = GetParser()
args = parser.parse_args()
EPOCHS, BATCH_SIZE = args.Epochs, args.Batch_Size


plot_train=False
out_of_sample=False




for fold_num in range(1,4):
    print(fold_num)
    data = data_builder(args, fold=fold_num, look_ahead=args.Look_Ahead, out_of_sample=out_of_sample)
    x_train, y_train, y_train_index, x_test, y_test, y_test_index = data.build(squared = args.Square_Inputs, normalise_all=True)

    input = tf.keras.layers.Input(shape=[x_train.shape[1], x_train.shape[2]])
    LSTM1 = tf.keras.layers.LSTM(x_train.shape[2], activation='relu', return_sequences=True)(input)
    LSTM2 = tf.keras.layers.LSTM(int(x_train.shape[2]/2), activation='relu', return_sequences=False)(LSTM1)
    Latent_Space = tf.keras.layers.Dense(64, activation='relu')(LSTM2)
    repeat = tf.keras.layers.RepeatVector(28)(Latent_Space)
    LSTM3 = tf.keras.layers.LSTM(int(x_train.shape[2] / 2), activation='relu', return_sequences=True)(repeat)
    output= tf.keras.layers.LSTM(x_train.shape[2], activation='linear', return_sequences=True)(LSTM3)
    model = tf.keras.Model(inputs=input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss = tf.keras.losses.mean_squared_error,
    )

    # model.load_weights('/Users/michael/Documents/datasets/weights/fold' + str(fold_num) + '_encoder_decoder.hdf5')
    model.fit(x_train, x_train,
              epochs=100,
              batch_size=32)
    model.save_weights('/Users/michael/Documents/datasets/weights/fold'+str(fold_num)+'_encoder_decoder.hdf5')



    input = tf.keras.layers.Input(shape=[x_train.shape[1], x_train.shape[2]])
    LSTM1 = tf.keras.layers.LSTM(x_train.shape[2], activation='relu', return_sequences=True)(input)
    LSTM2 = tf.keras.layers.LSTM(int(x_train.shape[2]/2), activation='relu', return_sequences=False)(LSTM1)
    Latent_Space = tf.keras.layers.Dense(64, activation='relu')(LSTM2)
    decoder = tf.keras.layers.Dense(10)
    output = tf.keras.layers.Dense(1)(Latent_Space)
    model2 = tf.keras.Model(inputs=input, outputs=output)

    for layer in range(1,4):
        model2.layers[layer].set_weights(model.layers[layer].get_weights())
        model2.layers[layer].trainable = False

    model2.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.mean_squared_error,
        metrics = ['mae', 'mse']
    )

    model2.fit(x_train, y_train[:,20],
              validation_data=(x_test, y_test[:, 20]),
              epochs=100,
              batch_size=32)

    model.save_weights('/Users/michael/Documents/datasets/weights/fold'+str(fold_num)+'_encoder_forecast.hdf5')

