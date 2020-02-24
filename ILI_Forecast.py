import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import metrics
from Parser import GetParser
from Functions import data_builder, build_attention, build_model, recurrent_attention, logger, simple, simple_GRU, plotter
from Transformer import encoder_network
from Time_Series_Transformer import transformer_network, modified_encoder
# tf.config.experimental_run_functions_eagerly(True)
import pandas as pd

confidence = False
parser = GetParser()
args = parser.parse_args()

EPOCHS, BATCH_SIZE = args.Epochs, args.Batch_Size

logging = logger(args)
models, look_aheads, max_k = logging.get_inputs()
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)
fig = plotter(1)

def validation(y_true, y_pred):
    mse = tf.abs(y_true - tf.squeeze(y_pred))

    mean1 = np.mean(mse[:365] / 3)
    mean2 = np.mean(mse[365:2 * 365])
    mean3 = np.mean(mse[2 * 365:])

    return mean1 + mean2 + mean3


class early_stopping:
    def __init__(self, patience):
        self.count = 1
        self.patience = patience

    def __call__(self, val_metric):
        if len(val_metric) > 0:
            if val_metric[epoch] > val_metric[epoch - self.count]:
                count = self.count + 1
                if count > self.patience:
                    self.count = 1
                    return True
            else:
                self.count = 1
                return False
do_early_stopping = False
fig = plotter(1)
loss = tf.keras.losses.MAE
for Model in models:
    for look_ahead in look_aheads:
        for k in range(max_k):
            for fold_num in range(1,5):
                print(k, fold_num)
                tf.random.set_seed(2)
                logging.update_details(fold_num=fold_num, k=k, model=Model, look_ahead=look_ahead)
                data = data_builder(args, fold=fold_num, look_ahead=look_ahead)
                x_train, y_train, y_train_index, x_test, y_test, y_test_index = data.build()

                if Model == 'simpleGRU':
                    model = simple_GRU(x_train, y_train)
                if Model == 'GRU':
                    model = build_model(x_train, y_train)

                    x_train = [x_train[:,:,-1, np.newaxis],x_train[:,:,:-1]]
                    x_test = [x_test[:, :, -1, np.newaxis], x_test[:, :, :-1]]

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

                elif Model == 'ATTENTION':
                    if args.Init =='uniform':
                        initializer = tf.keras.initializers.glorot_normal(seed=None)
                    else:
                        initializer = tf.keras.initializers.glorot_uniform(seed=None)

                    model = build_attention(x_train, y_train, num_heads=7, regularizer = args.Regularizer, initializer=initializer)

                elif Model == 'LINEAR':
                    model = simple(x_train)
                    y_train = y_train[:,-1]
                    y_test = y_test[:, -1]

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

                elif Model == 'MODENC':
                    model = modified_encoder(
                        output_size=y_train.shape[1],
                        num_layers=1,
                        units=x_train.shape[1],
                        d_model=x_train.shape[2],
                        num_heads=10,
                        dropout=0.1,
                        name="transformer")

                elif Model == 'GRU_BAYES':
                    def normal_scale_uncertainty(t, softplus_scale=0.5):
                        """Create distribution with variable mean and variance"""
                        return tfd.Normal(loc=t[..., :1],
                                          scale=1e-3 + tf.math.softplus(softplus_scale * t[..., 1:]))

                    confidence = True
                    tfd = tfp.distributions

                    loss = lambda y, p_y: -p_y.log_prob(y)



                    ili_input = tf.keras.layers.Input(shape=[x_train.shape[1], x_train.shape[2]])
                    x = tf.keras.layers.GRU(x_train.shape[1], activation='relu', return_sequences=True)(
                        ili_input)
                    x = tf.keras.layers.GRU(int((x_train.shape[2] - 1)), activation='relu', return_sequences=True)(x)
                    y = tf.keras.layers.GRU(int(0.75 * (x_train.shape[2] - 1)), activation='relu', return_sequences=False)(x)
                    z = tf.keras.layers.Dense(2, activation = 'relu')(y)
                    z = tfp.layers.DistributionLambda(normal_scale_uncertainty)(z)

                    model = tf.keras.models.Model(inputs=ili_input, outputs=z)

                    y_test = y_test[:, -1]
                    y_train = y_train[:, -1]

                elif Model == 'LINEAR_BAYES':

                    def normal_scale_uncertainty(t, softplus_scale=0.5):
                        """Create distribution with variable mean and variance"""
                        return tfd.Normal(loc=t[..., :1],
                                          scale=1e-3 + tf.math.softplus(softplus_scale * t[..., 1:]))

                    confidence = True
                    tfd = tfp.distributions

                    loss = lambda y, p_y: -p_y.log_prob(y)
                    # loss = lambda y, p_y: abs(y-p_y)
                    model = tf.keras.Sequential([
                        tf.keras.layers.Dense(1 + 1),
                        tfp.layers.DistributionLambda(normal_scale_uncertainty

                            # lambda t: tfd.Normal(loc=t[..., :1],
                            #                      scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]))
                        )
                    ])

                    x_train = x_train.reshape((x_train.shape[0], -1))
                    x_test = x_test.reshape((x_test.shape[0], -1))
                    y_test = y_test[:, -1]
                    y_train = y_train[:, -1]






                elif Model == 'R_ATTN':
                    model = recurrent_attention(x_train, y_train, num_heads=7, regularizer = args.Regularizer)
                    y_train = y_train[:,:,np.newaxis]
                    y_test = y_test[:,:,np.newaxis]

                if Model != 'TRANSFORMER':
                    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)
                    model.compile(optimizer=optimizer,
                                  loss=loss,
                                  metrics=['mae', 'mse', metrics.rmse])

                    if do_early_stopping:
                        x_train, y_train, x_val, y_val = data.split(x_train, y_train)
                        val_metric = []

                        patience = 5
                        for epoch in range(EPOCHS):
                            model.fit(
                                x_train, y_train,
                                epochs=1, batch_size=BATCH_SIZE)
                            val_metric.append(validation(y_val, model(x_val)))
                            print("validation loss = {:1.1f}".format(val_metric[epoch]))
                            if early_stopping(patience)(val_metric):
                                break
                        for j in range(2):
                            for epoch in range(EPOCHS):
                                x_train = np.append(x_train, x_val[:365], 0)
                                y_train = np.append(y_train, y_val[:365], 0)
                                x_val = x_val[365:]
                                y_val = y_val[365:]
                                model.fit(
                                    x_train, y_train,
                                    epochs=1, batch_size=BATCH_SIZE)
                                val_metric.append(validation(y_val, model(x_val)))
                                print("validation loss = {:1.1f}".format(val_metric[epoch]))
                                if early_stopping(patience)(val_metric):
                                    break

                        for epoch in range(5):
                            x_train = np.append(x_train, x_val, 0)
                            y_train = np.append(y_train, y_val, 0)
                            model.fit(
                                x_train, y_train,
                                epochs=1, batch_size=BATCH_SIZE)
                    else:
                        model.fit(
                            x_train, y_train,
                            epochs=EPOCHS, batch_size=BATCH_SIZE)

                if confidence:
                    yhat = model(x_test)
                    prediction = yhat.mean()
                    stddev = yhat.stddev()
                    fig.plot_conf(fold_num, prediction, y_test, stddev)
                else:
                    prediction = model.predict(x_test)
                    fig.plot(fold_num, prediction, y_test, x1=False)
                logging.log(prediction, y_test, model, save=True)

fig.save(logging.save_directory + '/predictions.png')
fig.show()

# final_weights = model.weights[0].numpy()[-167:]
# weights = pd.DataFrame(columns=['weight'],index = np.asarray(data.columns), data = np.squeeze(final_weights))
# weights.to_csv('weights.csv')
