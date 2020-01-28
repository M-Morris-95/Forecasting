import numpy as np
import tensorflow as tf
import metrics
from Parser import GetParser
from Functions import data_builder, build_attention, build_model, recurrent_attention, logger
from Transformer import encoder_network
from Time_Series_Transformer import transformer_network
parser = GetParser()
args = parser.parse_args()

EPOCHS, BATCH_SIZE = args.Epochs, args.Batch_Size

logging = logger(args)
models, look_aheads, max_k = logging.get_inputs()
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)

for Model in models:
    for look_ahead in look_aheads:
        for k in range(max_k):
            for fold_num in range(1,5):
                print(k, fold_num)
                tf.random.set_seed(k)
                logging.update_details(fold_num=fold_num, k=k, model=Model, look_ahead=look_ahead)
                data = data_builder(args, fold=fold_num, look_ahead=look_ahead)
                x_train, y_train, y_train_index, x_test, y_test, y_test_index = data.build()

                if Model == 'GRU':
                    model = build_model(x_train, y_train)

                    model.compile(optimizer=optimizer,
                                  loss='mse',
                                  metrics=['mae', 'mse', metrics.rmse])

                    model.fit(
                        [x_train[:,:,-1, np.newaxis],x_train[:,:,:-1]], y_train,
                        validation_data=([x_test[:,:,-1, np.newaxis],x_test[:,:,:-1]], y_test),
                        epochs=EPOCHS, batch_size=BATCH_SIZE)

                    prediction = model.predict([x_test[:, :, -1, np.newaxis], x_test[:, :, :-1]])

                elif Model == 'ENCODER':
                    x_train = np.swapaxes(x_train, 1, 2)
                    x_test = np.swapaxes(x_test, 1, 2)


                    model = encoder_network(
                        output_size=y_train.shape[1],
                        num_layers=2,
                        units=x_train.shape[1],
                        d_model=x_train.shape[2],
                        num_heads=7,
                        dropout=0.1,
                        name="encoder")

                    model.compile(optimizer=optimizer,
                                  loss='mse',
                                  metrics=['mae', 'mse', metrics.rmse])

                    model.fit(
                        x_train, y_train,
                        validation_data=(x_test, y_test),
                        epochs=EPOCHS, batch_size=BATCH_SIZE)

                    prediction = model(x_test, training=False)

                elif Model == 'ATTENTION':
                    model = build_attention(x_train, y_train, num_heads=7, regularizer = args.Regularizer)

                    model.compile(optimizer=optimizer,
                                  loss='mse',
                                  metrics=['mae', 'mse', metrics.rmse])

                    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logging.logging_directory, histogram_freq=1)

                    model.fit(
                        x_train, y_train,
                        validation_data=(x_test, y_test),
                        epochs=EPOCHS, batch_size=BATCH_SIZE)

                    prediction = model.predict(x_test)

                elif Model == 'TRANSFORMER':
                    model = transformer_network(
                        output_size=1,
                        num_layers=1,
                        units = x_train.shape[1],
                        d_model=x_train.shape[2],
                        num_heads=10,
                        dropout=0.1,
                        name="transformer")

                    model.compile(optimizer=optimizer,
                                  loss='mse',
                                  metrics=['mae', 'mse', metrics.rmse])

                    # inputs = none, 28, 180          None, 1

                    x_train2 = np.zeros((y_train.shape[0], y_train.shape[1]))
                    x_train2[:,1:] = y_train[:,:-1]
                    x_train2 = x_train2.reshape(-1)
                    x_train2 = x_train2[:, np.newaxis]
                    x_train = np.tile(x_train, [21, 1, 1])
                    y_train = y_train.reshape(-1)
                    y_train = y_train[:, np.newaxis]



                    model.fit(
                        [x_train, x_train2], y_train,
                        # validation_data=(x_test, y_test),
                        epochs=EPOCHS, batch_size=BATCH_SIZE)
                    prediction = np.zeros((x_test.shape[0], 22))

                    for i in range(x_test.shape[0]):
                        print(i)
                        for j in range(21):
                            temp = model([x_test[np.newaxis,i, :, :], prediction[np.newaxis,np.newaxis,i,j]], training=False)
                            prediction[i,j+1] = temp.numpy()

                elif Model == 'R_ATTN':
                    model = recurrent_attention(x_train, y_train, num_heads=7, regularizer = args.Regularizer)

                    model.compile(optimizer=optimizer,
                                  loss='mse',
                                  metrics=['mae', 'mse', metrics.rmse])

                    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logging.logging_directory, histogram_freq=1)

                    model.fit(
                        x_train, y_train[:,:,np.newaxis],
                        validation_data=(x_test, y_test[:,:,np.newaxis]),
                        epochs=EPOCHS, batch_size=BATCH_SIZE)

                    prediction = model.predict(x_test)

                logging.log(prediction, y_test, model, save=True)
