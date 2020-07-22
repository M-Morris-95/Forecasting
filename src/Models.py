import tensorflow as tf
import tensorflow_probability as tfp
from metrics import *
from Transformer import *
# from Time_Series_Transformer import *
import tqdm
import matplotlib.pyplot as plt

class Linear:
    def __init__(self, x_train, y_train, regulariser=False, optimiser=False, loss=False):
        self.train_prediction = None
        if regulariser:
            self.regulariser = tf.keras.regularizers.l2(0.01)
        else:
            self.regulariser = None

        if not optimiser:
            optimiser = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)

        if not loss:
            loss = tf.keras.losses.mean_squared_error

        ili_input = tf.keras.layers.Input(shape=[x_train.shape[1], x_train.shape[2]]) # because sliding window of shape n_daysxlagxn_features
        flatten = tf.keras.layers.Flatten()(ili_input) # need to be a vector
        output = tf.keras.layers.Dense(1)(flatten) # single neuron with linear activation
        self.model = tf.keras.models.Model(inputs=ili_input, outputs=output) # build model

        # compile, RMSprop and ADAM both good choices for optimiser
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9),
                           loss=tf.keras.losses.mean_squared_error,
                           )

        self.model.fit(x_train, y_train,
                       epochs=epochs,
                       batch_size=batch_size)

    def modify_data(self, x_train, y_train, x_test, y_test):
        y_train = y_train[:, -1]
        y_test = y_test[:, -1]
        return x_train, y_train, x_test, y_test

    def fit(self, x_train, y_train, epochs, batch_size, plot=False):
        self.model.fit(x_train, y_train,
                       epochs=epochs,
                       batch_size=batch_size)
        if plot:
            self.train_prediction, self.train_stddev, _, _ = self.predict(x_train)

    def predict(self, x_test):
        return self.model.predict(x_test), None, None, None

class Simple_GRU:
    def __init__(self, x_train, y_train, dropout = False, regulariser = False, optimiser = False, loss = False, mimo=False):
        self.train_prediction = None
        self.dropout=dropout
        if regulariser:
            self.regulariser = tf.keras.regularizers.l2(0.01)
        else:
            self.regulariser = None

        if not optimiser:
            optimiser = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)

        if not loss:
            loss = tf.keras.losses.mean_squared_error

        self.mimo=mimo

        ili_input = tf.keras.layers.Input(shape=[x_train.shape[1], x_train.shape[2]])
        x = tf.keras.layers.GRU(x_train.shape[1], activation='relu', return_sequences=True, kernel_regularizer=self.regulariser)(ili_input)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.GRU(int((x_train.shape[2] - 1)), activation='relu', return_sequences=True, kernel_regularizer=self.regulariser)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.GRU(int(0.75 * (x_train.shape[2] - 1)), activation='relu', return_sequences=True,kernel_regularizer=self.regulariser)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        if not mimo:
            z = tf.keras.layers.GRU(1, activation='linear', return_sequences=False, kernel_regularizer=self.regulariser)(x)
        else:
            z = tf.keras.layers.GRU(y_train.shape[1], activation='linear', return_sequences=False, kernel_regularizer=self.regulariser)(x)
        self.model = tf.keras.models.Model(inputs=ili_input, outputs=z)

        self.model.compile(optimizer=optimiser,
                      loss=loss,
                      metrics=['mae', 'mse'])


    def modify_data(self, x_train, y_train, x_test, y_test):
        if not self.mimo:
            y_train = y_train[:,-1]
        y_test = y_test[:,-1]
        return x_train, y_train, x_test, y_test

    def fit(self, x_train, y_train, epochs, batch_size, plot = False):
        self.model.fit(x_train, y_train,
                       epochs = epochs,
                       batch_size = batch_size)

        if plot:
            self.train_prediction, self.train_stddev, _,_ = self.predict(x_train)

    def predict(self, x_test):
        if not self.dropout:
            return self.model.predict(x_test)[:,-1], None, None, None

        else:
            preds = []
            for i in range(100):
                preds.append(self.model(x_test, training=True)[:, -1].numpy())

            preds = np.asarray(preds)
            pred_mean = np.mean(preds, 0)
            pred_std = np.std(preds, 0)
            return pred_mean, pred_std, None, None


class GRU:
    def __init__(self, x_train, y_train, regulariser=False, optimiser=False, loss=False):
        self.train_prediction = None
        if regulariser:
            self.regulariser = tf.keras.regularizers.l2(0.01)
        else:
            self.regulariser = None

        if not optimiser:
            optimiser = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)

        if not loss:
            loss = tf.keras.losses.mean_squared_error

        initializer = tf.keras.initializers.glorot_normal(seed=None)
        ili_input = tf.keras.layers.Input(shape=[x_train.shape[1], 1])

        x = tf.keras.layers.GRU(28, activation='relu', return_sequences=True, kernel_initializer=initializer)(ili_input)
        x = tf.keras.Model(inputs=ili_input, outputs=x)

        google_input = tf.keras.layers.Input(shape=[x_train.shape[1], x_train.shape[2] - 1])
        y = tf.keras.layers.GRU(x_train.shape[2] - 1, activation='relu', return_sequences=True, kernel_initializer=initializer)(
            google_input)
        y = tf.keras.layers.GRU(int(0.5 * (x_train.shape[2] - 1)), activation='relu', return_sequences=True,
                kernel_initializer=initializer)(y)
        y = tf.keras.Model(inputs=google_input, outputs=y)

        z = tf.keras.layers.concatenate([x.output, y.output])
        z = tf.keras.layers.GRU(y_train.shape[1], activation='relu', return_sequences=False, kernel_initializer=initializer)(z)

        self.model = tf.keras.Model(inputs=[x.input, y.input], outputs=z)

        self.model.compile(optimizer=optimiser,
                           loss=loss,
                           metrics=['mae', 'mse'])

    def modify_data(self, x_train, y_train, x_test, y_test):
        x_train = [x_train[:, :, -1, np.newaxis], x_train[:, :, :-1]]
        x_test = [x_test[:, :, -1, np.newaxis], x_test[:, :, :-1]]
        return x_train, y_train, x_test, y_test

    def fit(self, x_train, y_train, epochs, batch_size, plot=False):
        self.model.fit(x_train, y_train,
                       epochs=epochs,
                       batch_size=batch_size)

        if plot:
            self.train_prediction, self.train_stddev = self.predict(x_train)

    def predict(self, x_test):
        return self.model.predict(x_test), None

class Encoder:
    def __init__(self, x_train, y_train, regulariser=False, optimiser=False, loss=False):
        self.train_prediction = None
        if regulariser:
            self.regulariser = tf.keras.regularizers.l2(0.01)
        else:
            self.regulariser = None

        if not optimiser:
            optimiser = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)

        if not loss:
            loss = tf.keras.losses.mean_squared_error

        self.model = encoder_network(
            output_size=y_train.shape[1],
            num_layers=2,
            units=x_train.shape[1],
            d_model=x_train.shape[2],
            num_heads=7,
            dropout=0.1,
            name="encoder")

        self.model.compile(optimizer=optimiser,
                           loss=loss,
                           metrics=['mae', 'mse'])

    def modify_data(self, x_train, y_train, x_test, y_test):
        x_train = np.swapaxes(x_train, 1, 2)
        x_test = np.swapaxes(x_test, 1, 2)
        return x_train, y_train, x_test, y_test

    def fit(self, x_train, y_train, epochs, batch_size, plot=False):
        self.model.fit(x_train, y_train,
                       epochs=epochs,
                       batch_size=batch_size)

        if plot:
            self.train_prediction, self.train_stddev = self.predict(x_train)

    def predict(self, x_test):
        return self.model.predict(x_test), None

class Attention:
    def __init__(self, x_train, y_train, regulariser=False, optimiser=False, loss=False):
        self.train_prediction = None
        if regulariser:
            self.regulariser = tf.keras.regularizers.l2(0.01)
        else:
            self.regulariser = None

        if not optimiser:
            optimiser = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)

        if not loss:
            loss = tf.keras.losses.mean_squared_error

        self.initialiser = None

        #     initializer = tf.keras.initializers.glorot_normal(seed=None)
        # else:
        #     initializer = tf.keras.initializers.glorot_uniform(seed=None)

        num_heads = 7

        d_model = x_train.shape[1]

        ili_input = tf.keras.layers.Input(shape=[x_train.shape[1], x_train.shape[2]])
        x = tf.keras.layers.GRU(x_train.shape[1], activation='relu', return_sequences=True, kernel_regularizer=self.regulariser,
                kernel_initializer=self.initialiser)(ili_input)

        x = MultiHeadAttention(d_model, num_heads, name="attention", regularizer=self.regulariser, initializer=self.initialiser)({
            'query': x,
            'key': x,
            'value': x
        })
        x = tf.keras.layers.GRU(int((x_train.shape[2] - 1)), activation='relu', return_sequences=True, kernel_regularizer=self.regulariser,
                kernel_initializer=self.initialiser)(x)
        y = tf.keras.layers.GRU(int(0.75 * (x_train.shape[2] - 1)), activation='relu', return_sequences=True,
                kernel_regularizer=self.regulariser, kernel_initializer=self.initialiser)(x)
        z = tf.keras.layers.GRU(y_train.shape[1], activation='relu', return_sequences=False, kernel_regularizer=self.regulariser,
                kernel_initializer=self.initialiser)(y)

        self.model = tf.keras.models.Model(inputs=ili_input, outputs=z)

        self.model.compile(optimizer=optimiser,
                      loss=loss,
                      metrics=['mae', 'mse'])

    def modify_data(self, x_train, y_train, x_test, y_test):
        return x_train, y_train, x_test, y_test

    def fit(self, x_train, y_train, epochs, batch_size, plot=False):
        self.model.fit(x_train, y_train,
                       epochs=epochs,
                       batch_size=batch_size)

        if plot:
            self.train_prediction, self.train_stddev = self.predict(x_train)

    def predict(self, x_test):
        return self.model.predict(x_test), None

class Recurrent_Attention:
    def __init__(self, x_train, y_train, regulariser=False, optimiser=False, loss=False):
        self.train_prediction = None
        if regulariser:
            self.regulariser = tf.keras.regularizers.l2(0.01)
        else:
            self.regulariser = None

        if not optimiser:
            optimiser = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)

        if not loss:
            loss = tf.keras.losses.mean_squared_error

        self.initialiser = None

        #     initializer = tf.keras.initializers.glorot_normal(seed=None)
        # else:
        #     initializer = tf.keras.initializers.glorot_uniform(seed=None)

        num_heads = 7
        d_model = x_train.shape[1]

        ili_input = tf.keras.layers.Input(shape=[x_train.shape[1], x_train.shape[2]])
        x = tf.keras.layers.GRU(x_train.shape[1], activation='relu', return_sequences=True, kernel_regularizer=self.regularizer)(ili_input)

        x = MultiHeadAttention(d_model, num_heads, name="attention", regularizer=self.regularizer)({
            'query': x,
            'key': x,
            'value': x
        })
        x = tf.keras.layers.GRU(int((x_train.shape[2] - 1)), activation='relu', return_sequences=True, kernel_regularizer=self.regularizer)(
            x)
        y = tf.keras.layers.GRU(int(0.75 * (x_train.shape[2] - 1)), activation='relu', return_sequences=False,
                kernel_regularizer=self.regularizer)(x)
        y = tf.keras.layers.RepeatVector((y_train.shape[1]))(y)
        z = tf.keras.layers.GRU(1, activation='relu', return_sequences=True, kernel_regularizer=self.regularizer)(y)
        model = tf.keras.models.Model(inputs=ili_input, outputs=z)

        model.compile(optimizer=optimiser,
                      loss=loss,
                      metrics=['mae', 'mse'])

    def modify_data(self, x_train, y_train, x_test, y_test):
        y_train = y_train[:, :, np.newaxis]
        y_test = y_test[:, :, np.newaxis]
        return x_train, y_train, x_test, y_test

    def fit(self, x_train, y_train, epochs, batch_size, plot=False):
        self.model.fit(x_train, y_train,
                       epochs=epochs,
                       batch_size=batch_size)

        if plot:
            self.train_prediction, self.train_stddev = self.predict(x_train)

    def predict(self, x_test):
        return self.model.predict(x_test), None

class Transformer:
    def __init__(self, x_train, y_train, regulariser=False, optimiser=False, loss=False):
        self.train_prediction = None
        if regulariser:
            self.regulariser = tf.keras.regularizers.l2(0.01)
        else:
            self.regulariser = None

        if not optimiser:
            optimiser = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)

        if not loss:
            loss = tf.keras.losses.mean_squared_error

        self.model = transformer_network(
            output_size=1,
            num_layers=1,
            units=x_train.shape[1],
            d_model=x_train.shape[2],
            num_heads=10,
            dropout=0.1,
            name="transformer")

        self.model.compile(optimizer=optimiser,
                           loss=loss,
                           metrics=['mae', 'mse'])

    def modify_data(self, x_train, y_train, x_test, y_test):
        self.x_train2 = np.zeros((y_train.shape[0], y_train.shape[1]))
        self.x_train2[:, 1:] = y_train[:, :-1]
        self.x_train2 = self.x_train2.reshape(-1)
        self.x_train2 = self.x_train2[:, np.newaxis]
        x_train = np.tile(x_train, [21, 1, 1])
        y_train = y_train.reshape(-1)
        y_train = y_train[:, np.newaxis]
        return x_train, y_train, x_test, y_test

    def fit(self, x_train, y_train, epochs, batch_size, plot=False):
        self.model.fit(
            [x_train, self.x_train2], y_train,
            epochs=epochs, batch_size=batch_size)


        self.model.fit(x_train, y_train,
                       epochs=epochs,
                       batch_size=batch_size)

        if plot:
            self.train_prediction, self.train_stddev = self.predict(x_train)

    def predict(self, x_test):
        prediction = np.zeros((x_test.shape[0], 22))
        for i in range(x_test.shape[0]):
            print(i)
            for j in range(21):
                temp = self.model([x_test[np.newaxis, i, :, :], prediction[np.newaxis, np.newaxis, i, j]], training=False)
                prediction[i, j + 1] = temp.numpy()

        return prediction, None

class Linear_Data_Uncertainty:
    def __init__(self, x_train, y_train, regulariser=False, optimiser=False, loss=False):
        x_train = x_train.reshape((x_train.shape[0], -1))
        y_train = y_train[:, -1]
        self.train_prediction = None
        if regulariser:
            self.regulariser = tf.keras.regularizers.l2(0.01)
        else:
            self.regulariser = None

        if not optimiser:
            optimiser = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)

        if not loss:
            loss = lambda y, p_y: -p_y.log_prob(y)

        # c = np.log(np.expm1(0.25))
        def normal_scale_uncertainty(t):
            """Create distribution with variable mean and variance"""
            return tfp.distributions.Normal(loc=t[:, :1],
                                            scale=1e-5 + 0.1*tf.nn.softplus(10*t[:, 1:]))

        ili_input = tf.keras.layers.Input(shape=x_train.shape[1])
        mean = tf.keras.layers.Dense(1, activation='linear')(ili_input)
        std = tf.keras.layers.Dense(1, activation='linear')(ili_input)

        concat = tf.keras.layers.concatenate([mean, std])
        prob = tfp.layers.DistributionLambda(normal_scale_uncertainty)(concat)
        self.model = tf.keras.models.Model(inputs=ili_input, outputs=prob)

        self.model.compile(optimizer=optimiser,
                           loss=loss,
                           metrics=['mae', 'mse'])

    def modify_data(self, x_train, y_train, x_test, y_test):
        x_train = x_train.reshape((x_train.shape[0], -1))
        x_test = x_test.reshape((x_test.shape[0], -1))
        y_test = y_test[:, -1]
        y_train = y_train[:, -1]
        return x_train, y_train, x_test, y_test

    def fit(self, x_train, y_train, epochs, batch_size, plot=False):
        self.log = {'Loss': [],
                    'mae': [],
                    'mse': []}
        self.model.fit(x_train, y_train,
                       epochs=epochs,
                       batch_size=batch_size)
        self.log['Loss'].append(self.model.history.history['loss'])
        self.log['mae'].append(self.model.history.history['mae'])
        self.log['mse'].append(self.model.history.history['mse'])
        if plot:

            self.train_prediction, self.train_stddev, _, _ = self.predict(x_train)

    def predict(self, x_test):
        yhat = self.model(x_test)
        prediction = yhat.mean()
        stddev = yhat.stddev()
        return prediction, stddev, None, None

class GRU_Data_Uncertainty:
    def __init__(self, x_train, y_train, regulariser=False, optimiser=False, loss=False):
        # x_train = x_train.reshape((x_train.shape[0], -1))
        y_train = y_train[:, -1]
        tfd = tfp.distributions
        self.train_prediction = None
        if regulariser:
            self.regulariser = tf.keras.regularizers.l2(0.01)
        else:
            self.regulariser = None

        if not optimiser:
            optimiser = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)

        if not loss:
            loss = lambda y, p_y: -p_y.log_prob(y)

        def normal_scale_uncertainty(t):
            """Create distribution with variable mean and variance"""
            return tfp.distributions.Normal(loc=t[:, :1],
                                            scale=1e-5 + 0.1*tf.nn.softplus(10*t[:, 1:]))


        ili_input = tf.keras.layers.Input(shape=x_train.shape[1])
        mean = tf.keras.layers.Dense(1, activation='linear')(ili_input)
        std = tf.keras.layers.Dense(1, activation='linear')(ili_input)

        concat = tf.keras.layers.concatenate([mean, std])

        ili_input = tf.keras.layers.Input(shape=[x_train.shape[1], x_train.shape[2]])
        x = tf.keras.layers.GRU(x_train.shape[1], activation='relu', return_sequences=True)(
            ili_input)
        x = tf.keras.layers.GRU(int((x_train.shape[2] - 1)), activation='relu', return_sequences=True)(x)
        y = tf.keras.layers.GRU(int(0.75 * (x_train.shape[2] - 1)), activation='relu', return_sequences=False)(x)
        z = tf.keras.layers.Dense(2, activation='relu')(y)
        z = tfp.layers.DistributionLambda(normal_scale_uncertainty)(z)

        self.model = tf.keras.models.Model(inputs=ili_input, outputs=z)

        self.model.compile(optimizer=optimiser,
                           loss=loss,
                           metrics=['mae', 'mse'])

    def modify_data(self, x_train, y_train, x_test, y_test):
        y_test = y_test[:, -1]
        y_train = y_train[:, -1]
        return x_train, y_train, x_test, y_test

    def fit(self, x_train, y_train, epochs, batch_size, plot=False):
        self.model.fit(x_train, y_train,
                       epochs=epochs,
                       batch_size=batch_size)
        if plot:
            self.train_prediction, self.train_stddev = self.predict(x_train)

    def predict(self, x_test):
        yhat = self.model(x_test)
        prediction = yhat.mean()
        stddev = yhat.stddev()
        return prediction, stddev, None, None

class ARIMA:
    def __init__(self, x_train, y_train, args=None):
        from statsmodels.tsa.arima_model import ARIMA
        self.model = ARIMA(y_train[:,-1], order=(5,1,0))

    def modify_data(self, x_train, y_train, x_test, y_test):
        return x_train, y_train, x_test, y_test

    def fit(self, x_train, y_train, epochs, batch_size, plot=False):
        self.model.fit(disp=0)

    def predict(self):
        return 0

class GRU_Model_Uncertainty:
    def __init__(self, x_train, y_train, regulariser=False, optimiser=False, loss=False, args=None):
        KL_anneal = 1
        kl_loss_weight = KL_anneal * args.Batch_Size / x_train.shape[0]
        tfd = tfp.distributions
        self.train_prediction = None

        if not optimiser:
            optimiser = tf.keras.optimizers.Adam()

        loss = lambda y, p_y: -p_y.log_prob(y)

        tfd = tfp.distributions
        c = np.log(np.expm1(0.5))
        # Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
        def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size

            return tf.keras.Sequential([
                tfp.layers.VariableLayer(2 * n, dtype=dtype),
                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.Normal(loc=t[..., :n],
                               scale=1e-5 + tf.nn.softplus(c+t[..., n:])),
                    reinterpreted_batch_ndims=1)),
            ])

        # Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
        def prior_trainable(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            return tf.keras.Sequential([
                tfp.layers.VariableLayer(n, dtype=dtype),
                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.Normal(loc=t, scale=1),
                    reinterpreted_batch_ndims=1)),
            ])



        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[x_train.shape[1], x_train.shape[2]]),
            tf.keras.layers.GRU(x_train.shape[1], activation='relu', return_sequences=True),
            tf.keras.layers.GRU(40, activation='relu', return_sequences=False),
            # tf.keras.layers.GRU(20, activation='relu', return_sequences=False),

            tfp.layers.DenseVariational(units=20,
                                        activation='relu',
                                        make_posterior_fn=posterior_mean_field,
                                        make_prior_fn=prior_trainable,
                                        kl_weight=kl_loss_weight),

            tfp.layers.DenseVariational(units=1,
                                        make_posterior_fn=posterior_mean_field,
                                        make_prior_fn=prior_trainable,
                                        kl_weight=kl_loss_weight),

            tfp.layers.DistributionLambda(
                lambda t: tfd.Normal(loc=t[..., :1],
                                     scale=1)),
        ])


        self.model.compile(optimizer=optimiser,
                           loss=loss,
                           metrics=['mae', 'mse'])

    def modify_data(self, x_train, y_train, x_test, y_test):

        y_test = y_test[:, -1]
        y_train = y_train[:, -1]
        return x_train, y_train, x_test, y_test

    def fit(self, x_train, y_train, epochs, batch_size, plot=False):
        self.model.fit(x_train, y_train,
                       epochs=epochs,
                       batch_size=batch_size)
        if plot:
            self.train_prediction, self.train_stddev, _, _ = self.predict(x_train)

    def predict(self, x_test):
        yhats = [self.model(x_test).mean() for _ in range(25)]

        means = []
        for yhat in yhats:
            means.append(yhat)
        means = np.squeeze(np.asarray(means))

        stddev = np.std(means, 0)
        prediction = np.mean(means, 0)
        return prediction, stddev, None, None

class Linear_Model_Uncertainty:
    def __init__(self, x_train, y_train, regulariser=False, optimiser=False, loss=False, args = None):
        x_train = x_train.reshape((x_train.shape[0], -1))

        tfd = tfp.distributions
        self.train_prediction = None
        if regulariser:
            self.regulariser = tf.keras.regularizers.l2(0.01)
        else:
            self.regulariser = None

        if not optimiser:
            optimiser = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)

        if not loss:
            loss = lambda y, p_y: -p_y.log_prob(y)

        tfd = tfp.distributions

        # Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.

        def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            c = np.log(np.expm1(0.2))
            return tf.keras.Sequential([
                tfp.layers.VariableLayer(2 * n, dtype=dtype),
                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.Normal(loc=t[..., :n],
                               # scale=1e-5 + tf.nn.softplus(c+t[..., n:])),
                               scale=1e-5 + 0.1*tf.nn.softplus(10*t[..., n:])),
                    reinterpreted_batch_ndims=None))
            ])

        def prior_trainable(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size

            return tf.keras.Sequential([
                tfp.layers.VariableLayer(n, dtype=dtype),
                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.Normal(loc=t, scale=1),
                    # tfd.Normal(loc=t, scale=2.5),
                    reinterpreted_batch_ndims=None))
            ])

        self.kl_loss_weight = 1.0 / (x_train.shape[0] / args.Batch_Size)

        Input = tf.keras.layers.Input(shape=x_train.shape[1])
        DenseVariational = tfp.layers.DenseVariational(1, posterior_mean_field, prior_trainable, kl_weight=self.kl_loss_weight)(Input)
        DistributionLambda = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1))(DenseVariational)

        self.model = tf.keras.models.Model(inputs=Input, outputs=DistributionLambda)

        self.model.compile(optimizer=optimiser,
                           loss=loss,
                           metrics=['mae', 'mse'])

    def modify_data(self, x_train, y_train, x_test, y_test):
        x_train = x_train.reshape((x_train.shape[0], -1))
        x_test = x_test.reshape((x_test.shape[0], -1))
        y_test = y_test[:, -1]
        y_train = y_train[:, -1]
        return x_train, y_train, x_test, y_test

    def fit(self, x_train, y_train, epochs, batch_size, plot=False):
        self.log = {'Loss': [],
                    'mae': [],
                    'mse': [],
                    'KL':[]}
        for epoch in range(epochs):
            print(epoch)
            self.model.fit(x_train, y_train,
                           epochs=1,
                           batch_size=batch_size,
                           verbose = 0)


            activation = x_train
            KL = 0
            for layer in self.model.layers:
                try:
                    KL = KL + self.kl_loss_weight*np.sum(tfp.distributions.kl_divergence(
                        layer._posterior(activation),
                        layer._prior(activation)).numpy())
                except:
                    pass
                activation = layer(activation)
            self.log['KL'].append(KL)


            self.log['Loss'].append(self.model.history.history['loss'][-1])
            self.log['mae'].append(self.model.history.history['mae'][-1])
            self.log['mse'].append(self.model.history.history['mse'][-1])

        if plot:
            self.train_prediction, self.train_stddev,_,_ = self.predict(x_train)

    def predict(self, x_test):
        yhats = [self.model(x_test).mean() for _ in range(25)]

        means = []
        for yhat in yhats:
            means.append(yhat)
        means = np.squeeze(np.asarray(means))

        stddev = np.std(means, 0)
        prediction = np.mean(means, 0)
        return prediction, stddev, None, None

class Linear_Combined_Uncertainty:
    def __init__(self, x_train, y_train, regulariser=False, optimiser=False, loss=False, args=None):
        x_train = x_train.reshape((x_train.shape[0], -1))

        tfd = tfp.distributions
        self.train_prediction = None
        if regulariser:
            self.regulariser = tf.keras.regularizers.l2(0.01)
        else:
            self.regulariser = None

        if not optimiser:
            optimiser = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)

        if not loss:
            loss = lambda y, p_y: -p_y.log_prob(y)

        tfd = tfp.distributions

        # Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.

        def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            c = np.log(np.expm1(0.5))
            return tf.keras.Sequential([
                tfp.layers.VariableLayer(2 * n, dtype=dtype),
                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.Normal(loc=t[..., :n],
                               # scale=1e-5 + tf.nn.softplus(c+t[..., n:])),
                               scale=1e-5 + 0.1*tf.nn.softplus(10*t[..., n:])),
                    reinterpreted_batch_ndims=None))
            ])

        def prior_trainable(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size

            return tf.keras.Sequential([
                tfp.layers.VariableLayer(n, dtype=dtype),
                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.Normal(loc=t, scale=2.5),
                    # tfd.Normal(loc=t, scale=2.5),
                    reinterpreted_batch_ndims=None))
            ])

        self.kl_loss_weight = 1 / (x_train.shape[0] / args.Batch_Size)

        Input = tf.keras.layers.Input(shape=x_train.shape[1])
        DenseVariational = tfp.layers.DenseVariational(32, posterior_mean_field, prior_trainable,
                                                       kl_weight=self.kl_loss_weight, activation='relu')(Input)
        DenseVariational = tfp.layers.DenseVariational(2, posterior_mean_field, prior_trainable,
                                                       kl_weight=self.kl_loss_weight)(DenseVariational)
        DistributionLambda =  tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t[..., :1],
                                 scale=1e-5 + tf.nn.softplus(np.log(np.expm1(1))+0*t[:, 1:])))(DenseVariational)
                                 # scale=1e-5 + 0.01*tf.math.softplus(5 * t[..., 1:])))(DenseVariational)

        self.model = tf.keras.models.Model(inputs=Input, outputs=DistributionLambda)

        self.model.compile(optimizer=optimiser,
                           loss=loss,
                           metrics=['mae', 'mse'])

    def modify_data(self, x_train, y_train, x_test, y_test):
        x_train = x_train.reshape((x_train.shape[0], -1))
        x_test = x_test.reshape((x_test.shape[0], -1))
        y_test = y_test[:, -1]
        y_train = y_train[:, -1]
        return x_train, y_train, x_test, y_test

    def fit(self, x_train, y_train, epochs, batch_size, plot=False):
        self.log = {'Loss': [],
                    'mae': [],
                    'mse': [],
                    'KL': []}
        self.predict(x_train)
        for epoch in range(epochs):
            print(epoch)
            self.model.fit(x_train, y_train,
                           epochs=1,
                           batch_size=batch_size,
                           verbose=0)

            activation = x_train
            KL = 0
            for layer in self.model.layers:
                try:
                    KL = KL + self.kl_loss_weight * np.sum(tfp.distributions.kl_divergence(
                        layer._posterior(activation),
                        layer._prior(activation)).numpy())
                except:
                    pass
                activation = layer(activation)
            self.log['KL'].append(KL)

            self.log['Loss'].append(self.model.history.history['loss'][-1])
            self.log['mae'].append(self.model.history.history['mae'][-1])
            self.log['mse'].append(self.model.history.history['mse'][-1])

        if plot:
            self.train_prediction, self.train_stddev, _, _ = self.predict(x_train)

    def predict(self, x_test):
        num_poll = 50
        num_sample = 50
        yhats = [self.model(x_test) for _ in range(num_poll)]

        pred = []
        model_mean = []
        for yhat in tqdm.tqdm(yhats):
            for i in range(num_sample):
                pred.append(np.squeeze(yhat.sample().numpy()))
            model_mean.append(np.squeeze(yhat.mean().numpy()))
        pred = np.asarray(pred)
        model_mean = np.asarray(model_mean)

        stddev = np.std(pred, 0)
        prediction = np.mean(pred, 0)

        stddev = np.std(model_mean, 0)
        prediction = np.mean(model_mean, 0)

        return prediction, stddev, pred, model_mean

class GRU_Combined_Uncertainty:
    def __init__(self, x_train, y_train, optimiser=False, loss=False, args = None):
        KL_anneal = 2
        self.kl_loss_weight = KL_anneal * args.Batch_Size / x_train.shape[0]
        tfd = tfp.distributions

        self.train_prediction = None

        optimiser = tf.keras.optimizers.Adam()

        loss = lambda y, p_y: -p_y.log_prob(y)

        # Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
        def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            return tf.keras.Sequential([
                tfp.layers.VariableLayer(2 * n, dtype=dtype),
                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.Normal(loc=t[..., :n],
                               scale=1e-5 + 0.1 * tf.nn.softplus(10 * t[..., n:])),
                    reinterpreted_batch_ndims=1)),
            ])

        # Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
        def prior_trainable(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            return tf.keras.Sequential([
                tfp.layers.VariableLayer(n, dtype=dtype),
                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.Normal(loc=t, scale=5),
                    reinterpreted_batch_ndims=1)),
            ])

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[x_train.shape[1], x_train.shape[2]]),
            tf.keras.layers.GRU(x_train.shape[1], activation='relu', return_sequences=True),
            tf.keras.layers.GRU(40, activation='relu', return_sequences=True),
            tf.keras.layers.GRU(20, activation='relu', return_sequences=False),

            tfp.layers.DenseVariational(units=2*y_train.shape[1],
                                        make_posterior_fn=posterior_mean_field,
                                        make_prior_fn=prior_trainable,
                                        kl_weight=self.kl_loss_weight),

            tfp.layers.DistributionLambda(
                lambda t: tfd.Normal(loc=t[..., :y_train.shape[1]],
                                     scale=1e-5 + tf.math.softplus(t[..., y_train.shape[1]:]))),
        ])

        self.model.compile(optimizer=optimiser,
                           loss=loss,
                           metrics=['mae', 'mse'])



    def modify_data(self, x_train, y_train, x_test, y_test):
        return x_train, y_train, x_test, y_test

    def fit(self, x_train, y_train,  epochs, batch_size, plot=False):
        if plot:
            self.log = {'Likelihood_Loss': [],
                        'KL_Loss':[],
                        'Loss':[],
                        'mae':[],
                        'mse':[]}

            for _ in tqdm.tqdm(range(epochs)):
                self.model.fit(x_train, y_train,
                          epochs=1,
                          batch_size=batch_size,
                          verbose=0)

                KL = 0
                activation = x_train
                for layer in self.model.layers:
                    activation = layer(activation)

                    try:
                        KL = KL + np.sum(tfp.distributions.kl_divergence(
                            layer._posterior(activation),
                            layer._prior(activation)).numpy())
                    except:
                        pass

                Likelihood = np.mean(self.model.loss(y_train, activation))

                self.log['Likelihood_Loss'].append(Likelihood)
                self.log['KL_Loss'].append(KL)
                self.log['Loss'].append(self.model.history.history['loss'])
                self.log['mae'].append(self.model.history.history['mae'])
                self.log['mse'].append(self.model.history.history['mse'])

        else:
            self.log = {'Loss':[],
                        'mae':[],
                        'mse':[]}

            self.model.fit(x_train, y_train,
                           epochs = epochs,
                           batch_size=batch_size,
                           verbose = 1)

            self.log['Loss'].append(self.model.history.history['loss'])
            self.log['mae'].append(self.model.history.history['mae'])
            self.log['mse'].append(self.model.history.history['mse'])

    def plot(self, axes=1):
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('epoch')

        ax1.plot(self.lik_loss, label='likelihood', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylabel('likelihood', color=color)

        if axes == 2:
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

            color = 'tab:blue'
            ax2.set_ylabel('KL divergence', color=color)  # we already handled the x-label with ax1
            ax2.plot(self.kl_loss, label='KL divergence', color=color)
            ax2.tick_params(axis='y', labelcolor=color)

        else:
            # ax1.plot(self.kl_loss, label='KL divergence', color=color)
            ax1.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

        plt.legend()
        plt.show()

    def predict(self, x_test):
        num_poll = 50
        num_sample = 50
        yhats = [self.model(x_test) for _ in range(num_poll)]

        pred = []
        model_mean = []
        for yhat in tqdm.tqdm(yhats):
            for i in range(num_sample):
                pred.append(np.squeeze(yhat.sample().numpy()))
            model_mean.append(np.squeeze(yhat.mean().numpy()))
        pred = np.asarray(pred)
        model_mean = np.asarray(model_mean)

        stddev = np.std(pred, 0)
        prediction = np.mean(pred, 0)

        return prediction, stddev, pred, model_mean