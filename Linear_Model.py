import numpy as np
import tensorflow as tf
import metrics
from Parser import GetParser
from Functions import data_builder,logger, plotter
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from metrics import evaluate
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
# tf.config.experimental_run_functions_eagerly(True)
import pandas as pd
parser = GetParser()
args = parser.parse_args()

EPOCHS, BATCH_SIZE = args.Epochs, args.Batch_Size

logging = logger(args)
models, look_aheads, max_k = logging.get_inputs()
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9)
# optimizer = tf.keras.optimizers.Adam()
# optimizer = tf.keras.optimizers.SGD()
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.0005)
reg = np.zeros((16,2))
for idx, i in enumerate([0, 0.001, 0.01, 0.1]):
    for jdx, j in enumerate([0, 0.001, 0.01, 0.1]):
        reg[(idx + 1) * jdx + jdx, 0] = i
        reg[(idx + 1) * jdx + jdx, 1] = j

for Model in models:
    for look_ahead in look_aheads:
        for k in range(max_k):
            for fold_num in range(1,5):
                tf.random.set_seed(0)
                logging.update_details(fold_num=fold_num, k=k, model=Model, look_ahead=look_ahead)



                # regularizer = tf.keras.regularizers.L1L2(reg[k,0], reg[k,1])
                regularizer = None

                data = data_builder(args, fold=fold_num, look_ahead=look_ahead)
                x_train, y_train, y_train_index, x_test, y_test, y_test_index = data.build()

                x_train = x_train.reshape((x_train.shape[0], -1))
                x_test = x_test.reshape((x_test.shape[0], -1))
                y_train = y_train[:, -1]
                y_test = y_test[:, -1]

                ili_input = tf.keras.layers.Input(shape=[x_train.shape[1]])
                # hidden = tf.keras.layers.Dense(100)(ili_input)
                # hidden2 = tf.keras.layers.Dense(50)(hidden)
                output = tf.keras.layers.Dense(1)(ili_input)
                model = tf.keras.models.Model(inputs=ili_input, outputs=output)

                model.compile(optimizer=optimizer,
                              loss='mae',
                              metrics=['mae', 'mse', metrics.rmse])

                model.fit(
                    x_train, y_train,
                    epochs=EPOCHS, batch_size=BATCH_SIZE,
                    verbose = 2)

                # reg = SGDRegressor().fit(x_train, y_train)


                prediction = model.predict(x_test)
                plt.subplot(2, 2, fold_num)
                plt.plot(prediction)
                plt.plot(y_test)
                # plt.plot(reg.predict(x_test))
                logging.log(prediction, y_test, model, save=True)
                # print(evaluate(y_test,reg.predict(x_test) ))
plt.show()
print(logging.train_stats)

# final_weights = model.weights[0].numpy()[-167:]
# weights = pd.DataFrame(columns=['weight'],index = np.asarray(data.columns), data = np.squeeze(final_weights))
# weights.to_csv('weights.csv')

# plot1 = plotter(1)
# plot1.plot_df(logging)

# for i in range(len(reg)):
#     print('\'''L1=', reg[i, 0], 'L2=', reg[i, 1], '\',', end='')
