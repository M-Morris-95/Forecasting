import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from Parser import GetParser
from Functions import *
from Transformer import *
from Time_Series_Transformer import *
from Data_Builder import *
from Plotter import *
from Logger import *
from Early_Stopping import *



confidence = False
Ensemble = False
parser = GetParser()
args = parser.parse_args()

EPOCHS, BATCH_SIZE = args.Epochs, args.Batch_Size

logging = logger(args)
models, look_aheads, max_k = logging.get_inputs()
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)
fig = plotter(1)

fig2 = [plotter(2, size=[20,5], dpi=500),  plotter(3, size=[12,10], dpi=500),  plotter(4, size=[12,10], dpi=500),  plotter(5, size=[12,10], dpi=500)]
# fig2 = plotter(2, size=[12,10])
plot_train=False

do_early_stopping = False

loss = tf.keras.losses.MAE
for Model in models:
    for look_ahead in look_aheads:
        for k in range(max_k):
            for fold_num in range(1,5):
                print(k, fold_num)
                tf.random.set_seed(0)
                logging.update_details(fold_num=fold_num, k=k, model=Model, look_ahead=look_ahead)
                data = data_builder(args, fold=fold_num, look_ahead=look_ahead)
                x_train, y_train, y_train_index, x_test, y_test, y_test_index = data.build(squared = args.Square_Inputs, normalise_all=True)

                if Model == 'simpleGRU':
                    model = simple_GRU(x_train, y_train)

                if Model == 'GRU':
                    model = full_GRU(x_train, y_train)

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
                        # return tfd.Normal(loc=t[..., :1],
                        #                   scale=1e-3 + tf.math.softplus(softplus_scale * t[..., 1:]))
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

                elif Model == 'LINEAR_DATA_UNCERTAINTY':
                    x_train = x_train.reshape((x_train.shape[0], -1))
                    x_test = x_test.reshape((x_test.shape[0], -1))
                    y_test = y_test[:, -1]
                    y_train = y_train[:, -1]
                    confidence = True
                    loss = lambda y, p_y: -p_y.log_prob(y)

                    def normal_scale_uncertainty(t):
                        """Create distribution with variable mean and variance"""
                        return tfp.distributions.Normal(loc=t[:, :1],
                                          scale=1e-5+(t[:, 1:]))

                    ili_input = tf.keras.layers.Input(shape=x_train.shape[1])
                    mean = tf.keras.layers.Dense(1, activation='linear')(ili_input)
                    std = tf.keras.layers.Dense(1, activation = 'linear')(ili_input)

                    concat = tf.keras.layers.concatenate([mean, std])
                    prob = tfp.layers.DistributionLambda(normal_scale_uncertainty)(concat)
                    model = tf.keras.models.Model(inputs=ili_input, outputs=prob)

                elif Model == 'LINEAR_MODEL_UNCERTAINTY':
                    x_train = x_train.reshape((x_train.shape[0], -1))
                    x_test = x_test.reshape((x_test.shape[0], -1))
                    y_test = y_test[:, -1]
                    y_train = y_train[:, -1]
                    Ensemble = True
                    tfd = tfp.distributions

                    def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
                        n = kernel_size + bias_size

                        return tf.keras.Sequential([
                            tfp.layers.VariableLayer(2 * n, dtype=dtype),
                            tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                                tfd.Normal(loc=t[..., :n],
                                           scale=1e-5 + 0.2*tf.nn.softplus(t[..., n:])),
                                reinterpreted_batch_ndims=None))
                        ])
                        '''reinterpreted_batch_ndims: Scalar, integer number of rightmost batch dims which will 
                        be regarded as event dims. When None all but the first batch axis (batch axis 0) will be 
                        transferred to event dimensions (analogous to tf.layers.flatten).'''



                    def prior_trainable(kernel_size, bias_size=0, dtype=None):
                        n = kernel_size + bias_size

                        return tf.keras.Sequential([
                            tfp.layers.VariableLayer(n, dtype=dtype),
                            tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                                tfd.Normal(loc=t, scale=2.5),
                                reinterpreted_batch_ndims=None))
                        ])

                    Input = tf.keras.layers.Input(shape=x_train.shape[1])

                    # Dense = tf.keras.layers.Dense(10)(Input)
                    DenseVariational = tfp.layers.DenseVariational(1, posterior_mean_field, prior_trainable)(Input)
                    DistributionLambda = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1))(DenseVariational)

                    model = tf.keras.models.Model(inputs=Input, outputs=DenseVariational)

                elif Model == 'GRU_MODEL_UNCERTAINTY':
                    y_test = y_test[:, -1]
                    y_train = y_train[:, -1]
                    Ensemble = True
                    tfd = tfp.distributions

                    def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
                        n = kernel_size + bias_size

                        return tf.keras.Sequential([
                            tfp.layers.VariableLayer(2 * n, dtype=dtype),
                            tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                                tfd.Normal(loc=t[..., :n],
                                           scale=1e-5 + tf.nn.softplus(t[..., n:])),
                                reinterpreted_batch_ndims=None))
                        ])
                        '''reinterpreted_batch_ndims: Scalar, integer number of rightmost batch dims which will 
                        be regarded as event dims. When None all but the first batch axis (batch axis 0) will be 
                        transferred to event dimensions (analogous to tf.layers.flatten).'''


                    def prior_trainable(kernel_size, bias_size=0, dtype=None):
                        n = kernel_size + bias_size

                        return tf.keras.Sequential([
                            tfp.layers.VariableLayer(n, dtype=dtype, trainable=False),
                            tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                                tfd.Normal(loc=t, scale=5),
                                reinterpreted_batch_ndims=None))
                        ])

                    ili_input = tf.keras.layers.Input(shape=[x_train.shape[1], x_train.shape[2]])
                    GRU1 = tf.keras.layers.GRU(x_train.shape[1], activation='relu', return_sequences=True)(ili_input)
                    GRU2 = tf.keras.layers.GRU(int((x_train.shape[2] - 1)), activation='relu', return_sequences=True)(GRU1)
                    GRU3 = tf.keras.layers.GRU(int(0.75 * (x_train.shape[2] - 1)), activation='relu', return_sequences=False)(GRU2)
                    DenseVariational = tfp.layers.DenseVariational(1, make_posterior_fn=posterior_mean_field, make_prior_fn=prior_trainable)(GRU3)

                    model = tf.keras.models.Model(inputs=ili_input, outputs=DenseVariational)

                elif Model == 'FULL_GRU_MODEL_UNCERTAINTY':
                    y_test = y_test[:, -1]
                    y_train = y_train[:, -1]
                    Ensemble = True
                    tfd = tfp.distributions

                    def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
                        n = kernel_size + bias_size

                        return tf.keras.Sequential([
                            tfp.layers.VariableLayer(2 * n, dtype=dtype),
                            tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                                tfd.Normal(loc=t[..., :n],
                                           scale=1e-5 + tf.nn.softplus(t[..., n:])),
                                reinterpreted_batch_ndims=None))
                        ])

                    # def prior_trainable(kernel_size, bias_size=0, dtype=None):
                    #     n = kernel_size + bias_size
                    #
                    #     return tf.keras.Sequential([
                    #         tfp.layers.VariableLayer(n, dtype=dtype, trainable=False),
                    #         tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    #             tfd.Normal(loc=t,
                    #                        scale=2.5),
                    #             reinterpreted_batch_ndims=None))
                    #     ])


                    def prior_trainable(kernel_size, bias_size=0, dtype=None):
                        n = kernel_size + bias_size
                        return tf.keras.Sequential([
                            tfp.layers.VariableLayer(n, dtype=dtype),
                            tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                                tfd.Normal(loc=t, scale=1),
                                reinterpreted_batch_ndims=1)),
                        ])

                    ili_input = tf.keras.layers.Input(shape=[x_train.shape[1], 1])

                    x = tf.keras.layers.GRU(28, activation='relu', return_sequences=True)(ili_input)
                    x = tf.keras.models.Model(inputs=ili_input, outputs=x)

                    google_input = tf.keras.layers.Input(shape=[x_train.shape[1], x_train.shape[2] - 1])
                    y = tf.keras.layers.GRU(x_train.shape[2] - 1, activation='relu', return_sequences=True)(google_input)
                    y = tf.keras.layers.GRU(int(0.5 * (x_train.shape[2] - 1)), activation='relu', return_sequences=True)(y)
                    y = tf.keras.models.Model(inputs=google_input, outputs=y)

                    concatenate = tf.keras.layers.concatenate([x.output, y.output])
                    GRU1 = tf.keras.layers.GRU(int((concatenate.shape[2] - 1)), activation='relu', return_sequences=True)(concatenate)
                    GRU2 = tf.keras.layers.GRU(int(0.75 * (concatenate.shape[2] - 1)), activation='relu', return_sequences=False)(GRU1)
                    DenseVariational = tfp.layers.DenseVariational(1, make_posterior_fn=posterior_mean_field,make_prior_fn=prior_trainable)(GRU2)

                    model = tf.keras.models.Model(inputs=[x.input, y.input], outputs=DenseVariational)

                    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)

                    model.compile(optimizer=optimizer,
                                  loss='mae',
                                  metrics=['mae', 'mse', metrics.rmse])

                    x_train = [x_train[:,:,-1, np.newaxis],x_train[:,:,:-1]]
                    x_test = [x_test[:, :, -1, np.newaxis], x_test[:, :, :-1]]

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
                    if plot_train:
                        yhat = model(x_train)
                        prediction = yhat.mean()
                        stddev = yhat.stddev()
                        fig2[fold_num-1].plot_conf(fold_num, prediction, y_train, stddev, split=False)

                    yhat = model(x_test)
                    prediction = yhat.mean()
                    stddev = yhat.stddev()
                    fig.plot_conf(fold_num, prediction, y_test, stddev)

                elif Ensemble:
                    stddev = None
                    yhats = [model(x_test) for i in range(25)]

                    prediction = fig.plot_ensemble(fold_num, yhats, y_test)

                else:
                    if plot_train:
                        prediction = model.predict(x_train)
                        fig2[fold_num-1].plot(fold_num, prediction, y_train, x1=False, split=False)


                    stddev = None
                    prediction = model.predict(x_test)
                    fig.plot(fold_num, prediction, y_test, x1=False)


                if args.Logging:
                    logging.log(prediction, y_test, model, stddev, save=True, save_weights=False, col_names = data.columns)
if args.Logging:
    logging.save(last=True)
    fig.save(logging.save_directory + '/predictions.png')
fig.show()
for i in range(4):
    fig2[i].show()


# weights0 = model.weights[0].numpy()[:,0]
# weights1 = model.weights[0].numpy()[:,1]
# plt.plot(np.matmul(x_test, weights0) + np.matmul(x_test, weights1))
# plt.plot(np.matmul(x_test, weights0) - np.matmul(x_test, weights1))
# plt.plot(np.matmul(x_test, weights0))
# plt.show()