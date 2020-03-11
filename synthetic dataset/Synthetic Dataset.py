import numpy as np
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import tensorflow as tf
from Plotter import *
from make_data import *
from distributions import *
tfd = tfp.distributions

fig = plotter(1)

EPOCHS = 1000
length = 1028

x_train, y_train, x_test, y_test = get_data(length = length, stddev = 0.2, width = 1)
x_tst = x_train

negloglik = lambda y, p_y: -p_y.log_prob(y)

model = tf.keras.Sequential([
    tfp.layers.DenseVariational(1, posterior_mean_field, prior_trainable),
    tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=negloglik)

model.fit(x_train, y_train,
                epochs=500, batch_size=32)

# Make predictions.
yhat = [model(x_tst).mean() for i in range(100)]
yhat = np.asarray(yhat)

mean = np.squeeze(np.mean(yhat, 0))
stddev = np.squeeze(np.std(yhat, 0))

plt.scatter(x_train, y_train)
for i in range(yhat.shape[0]):
    plt.plot(np.squeeze(x_tst), yhat[i, :, :], linewidth=0.5, alpha=0.5, color='red')
plt.plot(x_tst, mean, color='red')
plt.show()