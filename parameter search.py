import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from Parser import GetParser
from Data_Builder import *
from Plotter import *
from Logger import *
from Early_Stopping import *

tfd = tfp.distributions
parser = GetParser()
args = parser.parse_args()

EPOCHS, BATCH_SIZE = args.Epochs, args.Batch_Size

logging = logger(args)

models, look_aheads, max_k = logging.get_inputs()

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)

A = [0.2, 0.5, 0.75, 1, 3, 5]
B = [0.2, 0.5, 0.75, 1, 3, 5]
C = [0.2, 0.5, 0.75, 1, 3, 5]
D = [0.2, 0.5, 0.75, 1, 3, 5]
k = 0
logging.iter=True


for posterior_mean_scaler in A:
    for posterior_std_scaler in B:
        for prior_mean_scaler in C:
            for prior_std_scaler in D:
                for look_ahead in look_aheads:
                    k = k + 1
                    for fold_num in range(1,5):
                        tf.random.set_seed(0)

                        logging.update_details(fold_num=fold_num, k=k, model='GRU_VI_Par_Search', look_ahead=look_ahead)
                        data = data_builder(args, fold=fold_num, look_ahead=look_ahead)
                        x_train, y_train, y_train_index, x_test, y_test, y_test_index = data.build(squared = args.Square_Inputs, normalise_all=True)
                        y_test = y_test[:, -1]
                        y_train = y_train[:, -1]

                        ''' MAKE MODEL '''


                        def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
                            n = kernel_size + bias_size
                            c = np.log(np.expm1(1.))
                            return tf.keras.Sequential([
                                tfp.layers.VariableLayer(2 * n, dtype=dtype),
                                tfp.layers.DistributionLambda(
                                    lambda t: tfd.Independent(  # pylint: disable=g-long-lambda
                                        tfd.Normal(loc=posterior_mean_scaler * t[..., :n],
                                                   scale=1e-5 + tf.nn.softplus(c + posterior_std_scaler * t[..., n:])),
                                        reinterpreted_batch_ndims=1)),
                            ])


                        def prior_trainable(kernel_size, bias_size=0, dtype=None):
                            n = kernel_size + bias_size
                            return tf.keras.Sequential([
                                tfp.layers.VariableLayer(n, dtype=dtype),
                                tfp.layers.DistributionLambda(
                                    lambda t: tfd.Independent(tfd.Normal(loc=prior_mean_scaler * t, scale=1),
                                                              # pylint: disable=g-long-lambda
                                                              reinterpreted_batch_ndims=prior_std_scaler)),
                            ])


                        loss = lambda y, p_y: -p_y.log_prob(y)

                        ili_input = tf.keras.layers.Input(shape=[x_train.shape[1], x_train.shape[2]])
                        GRU1 = tf.keras.layers.GRU(x_train.shape[1], activation='relu', return_sequences=True)(
                            ili_input)
                        GRU2 = tf.keras.layers.GRU(int((x_train.shape[2] - 1)), activation='relu',
                                                   return_sequences=True)(
                            GRU1)
                        GRU3 = tf.keras.layers.GRU(int(0.75 * (x_train.shape[2] - 1)), activation='relu',
                                                   return_sequences=False)(GRU2)
                        DenseVariational = tfp.layers.DenseVariational(1, make_posterior_fn=posterior_mean_field,
                                                                       make_prior_fn=prior_trainable)(GRU3)
                        DistributionLambda = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1))(
                            DenseVariational)
                        model = tf.keras.models.Model(inputs=ili_input, outputs=DistributionLambda)

                        model.compile(optimizer=optimizer,
                                      loss=loss,
                                      metrics=['mae', 'mse', metrics.rmse])

                        ''' TRAIN '''
                        for i in range(EPOCHS):
                            stddev = args.Noise_Std
                            model.fit(
                                x_train + np.random.normal(0, stddev, x_train.shape), y_train,
                                epochs=1,
                                batch_size=BATCH_SIZE)

                        yhats = np.squeeze(np.asarray([model(x_test).mean().numpy() for i in range(25)]))

                        prediction = np.mean(yhats, 0)
                        std = np.std(yhats, 0)

                        if args.Logging:
                            logging.log(y_pred = prediction, y_true = y_test, model = model, stddev = std, save=True, save_weights=False, col_names = data.columns)

logging.save(last=True)






