import pandas as pd
import os
import numpy as np
import tensorflow as tf
import time

from tensorflow.keras.callbacks import EarlyStopping

from Parser import GetParser
from Functions import plotter, build_model, rmse, evaluate, build_data, build_attention
from Tranformer import encoder_network


parser = GetParser()
args = parser.parse_args()

timestamp = time.strftime('%b-%d-%H-%M', time.localtime())

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

EPOCHS = 100
BATCH_SIZE = 64

save_dir = logging_dir + args.Model + timestamp
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
os.chdir(save_dir)

for fold_num in range(1,5):
    x_train, y_train, y_train_index, x_test, y_test, y_test_index  = build_data(data_dir + str(fold_num) + '/')

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)
    earlystop_callback = EarlyStopping(
        monitor='val_loss', min_delta=0.0001,
        patience=5)

    if args.Model == 'GRU':
        model = build_model(x_train)

        model.compile(optimizer=optimizer,
                      loss='mse',
                      metrics=['mae', 'mse', rmse])

        model.fit(
            [x_train[:,:,-1, np.newaxis],x_train[:,:,:-1]], y_train,
            callbacks=[earlystop_callback],
            validation_data=([x_test[:,:,-1, np.newaxis],x_test[:,:,:-1]], y_test),
            epochs=EPOCHS, batch_size=BATCH_SIZE)

        prediction = model.predict([x_test[:, :, -1, np.newaxis], x_test[:, :, :-1]])[:, 20]

    elif args.Model == 'ENCODER':

        model = encoder_network(
            output_size=y_train.shape[1],
            num_layers=3,
            units=x_train.shape[1],
            d_model=x_train.shape[2],
            num_heads=num_heads[fold_num],
            dropout=0.1,
            name="encoder")

        model.compile(optimizer=optimizer,
                      loss='mse',
                      metrics=['mae', 'mse', rmse])

        model.fit(
            x_train, y_train,
            callbacks=[earlystop_callback],
            validation_data=(x_test, y_test),
            epochs=EPOCHS, batch_size=BATCH_SIZE)

        prediction = model(x_test, training=False)[:, 20]

    elif args.Model == 'ATTENTION':

        model = build_attention(x_train, fold_num)

        model.compile(optimizer=optimizer,
                      loss='mse',
                      metrics=['mae', 'mse', rmse])

        model.fit(
            x_train, y_train,
            callbacks=[earlystop_callback],
            validation_data=(x_test, y_test),
            epochs=EPOCHS, batch_size=BATCH_SIZE)
        prediction = model.predict(x_test)[:, 20]

    y_test = y_test[:, -1]

    results[str(2014) + '/' + str(14 + fold_num)] = evaluate(y_test, prediction)
    test_predictions['prediction_'+str(2014) + '/' + str(14 + fold_num)] = prediction[:365]
    test_predictions['truth_' + str(2014) + '/' + str(14 + fold_num)] = y_test[:365]


    # model.save_weights('Fold_'+str(fold_num)+'network.hdf5')

    training_stats = pd.DataFrame(model.history.history)
    training_stats.to_csv(r'Fold_'+str(fold_num)+'_training_stats.csv')

    fig1.plot(fold_num, training_stats.mae, training_stats.val_mae)
    fig3.plot(fold_num, prediction, y_test, y_test_index)


results.to_csv(r'stats.csv')
test_predictions.to_csv(r'test_predictions.csv')

fig1.save('training_stats.png')
fig3.save('validation_predictions.png')

fig1.show()
fig3.show()
