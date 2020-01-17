import pandas as pd
import os
import numpy as np
import tensorflow as tf
import time

from tensorflow.keras.callbacks import EarlyStopping
import metrics
from Parser import GetParser
from Functions import plotter, data_builder, build_attention, build_model
from Tranformer import encoder_network


parser = GetParser()
args = parser.parse_args()


timestamp = time.strftime('%b-%d-%H-%M', time.localtime())

fig1 = plotter(1)
fig3 = plotter(3)

results = pd.DataFrame(index = ['MAE', 'RMSE', 'R'])
test_predictions = pd.DataFrame()


EPOCHS = 50
BATCH_SIZE = 128

use_day_of_the_year = True
if use_day_of_the_year:
    num_heads = [1,3,1,4,2]
else:
    num_heads = [1, 8, 4, 9, 11]

if not args.Server:
    logging_dir = '/Users/michael/Documents/github/Forecasting/Logging/'
else:
    logging_dir = '/home/mimorris/Forecasting/Logging/'

save_dir = logging_dir + 'ALL' + timestamp
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
os.chdir(save_dir)

if args.Model == 'ALL':
    Model = ['GRU', 'ATTENTION','ENCODER']
else:
    Model = [args.Model]

if args.Look_Ahead == 'ALL':
    look_ahead = [7, 14, 21]
else:
    look_ahead = [args.Look_Ahead]

for Model in Model:
    for look_ahead in look_ahead:
        for fold_num in range(1,5):
            data = data_builder(args, fold=fold_num, look_ahead=look_ahead, lag = args.Lag, country = args.Country)
            x_train, y_train, y_train_index, x_test, y_test, y_test_index  = data.build()

            optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)
            earlystop_callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5)

            if Model == 'GRU':
                model = build_model(x_train, y_train)

                model.compile(optimizer=optimizer,
                              loss='mse',
                              metrics=['mae', 'mse', metrics.rmse])

                model.fit(
                    [x_train[:,:,-1, np.newaxis],x_train[:,:,:-1]], y_train,
                    # callbacks=[earlystop_callback],
                    validation_data=([x_test[:,:,-1, np.newaxis],x_test[:,:,:-1]], y_test),
                    epochs=EPOCHS, batch_size=BATCH_SIZE)

                prediction = model.predict([x_test[:, :, -1, np.newaxis], x_test[:, :, :-1]])[:, -1]

            elif Model == 'ENCODER':
                num_heads = 7
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
                              metrics=['mae', 'mse', metrics.rmse])

                model.fit(
                    x_train, y_train,
                    # callbacks=[earlystop_callback],
                    validation_data=(x_test, y_test),
                    epochs=EPOCHS, batch_size=BATCH_SIZE)

                prediction = model(x_test, training=False)[:, -1]

            elif Model == 'ATTENTION':
                model = build_attention(x_train, y_train, num_heads=7) #num_heads=num_heads[fold_num])

                model.compile(optimizer=optimizer,
                              loss='mse',
                              metrics=['mae', 'mse', metrics.rmse])

                model.fit(
                    x_train, y_train,
                    # callbacks=[earlystop_callback],
                    validation_data=(x_test, y_test),
                    epochs=EPOCHS, batch_size=BATCH_SIZE)
                prediction = model.predict(x_test)[:, -1]

            y_test = y_test[:, -1]

            results[str(Model) +'_' + str(look_ahead) + '_' + str(2014) + '/' + str(14 + fold_num)] = metrics.evaluate(y_test, prediction)
            test_predictions[str(Model) +'_' + str(look_ahead) + '_' + 'prediction_'+str(2014) + '/' + str(14 + fold_num)] = prediction[:365]
            test_predictions['truth_' + str(2014) + '/' + str(14 + fold_num)] = y_test[:365]


            training_stats = pd.DataFrame(model.history.history)
            training_stats.to_csv(r'Fold_'+str(fold_num)+'_training_stats.csv')

            # fig1.plot(fold_num, training_stats.mae, training_stats.val_mae)
            # fig3.plot(fold_num, prediction, y_test, y_test_index)


        results.to_csv(r'stats.csv')
        test_predictions.to_csv(r'test_predictions.csv')

        # fig1.save('training_stats.png')
        # fig3.save('validation_predictions.png')
        #
        # fig1.show()
        # fig3.show()
