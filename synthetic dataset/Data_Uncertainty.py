import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import tensorflow as tf
from make_data import *
tf.config.experimental_run_functions_eagerly()

tfd = tfp.distributions

xlim=[0, 5]
ylim=[0, 1]
split = 0.25
length = 500
batch_size = 32
epochs = 500

# get data
x_train, y_train, x_test, y_test, y_stddev, y_mean = get_data(length=length, width=1, mean=0, stddev=0.8, split=split,
                                                      xlim=xlim, ylim=ylim)

# build model
negloglik = lambda y, p_y: -p_y.log_prob(y)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(2),
    tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1],
                             scale=1e-3 + tf.math.abs(t[..., 1:]))),
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=negloglik)

model.fit(x_train, y_train,
          epochs=epochs, batch_size=batch_size)

# predict train
yhat = model(x_train)
mean = yhat.mean()
stddev = yhat.stddev()

# plot train
plt.scatter(x_train, y_train, alpha=0.5, color='orange')
plt.plot(x_train, mean, color='green', linewidth=2)
plt.plot(x_train, mean + stddev, color='green', linewidth=1)
plt.plot(x_train, mean - stddev, color='green', linewidth=1)

# predict test
yhat = model(x_test)
mean = yhat.mean()
stddev = yhat.stddev()

# plot test
plt.scatter(x_test, y_test, alpha=0.5, color='orange')
plt.plot(x_test, mean, color='green', linewidth=2)
plt.plot(x_test, mean + stddev, color='green', linewidth=1)
plt.plot(x_test, mean - stddev, color='green', linewidth=1)

# tidy up graph
plt.plot([xlim[1]*(1-split), xlim[1]*(1-split)], [-100, 100], color = 'black')
plt.ylim((min(y_train), max(y_train)))
plt.annotate('train', (xlim[1]*(1-split)/2,max(y_train)*0.9))
plt.annotate('test', (xlim[1] - xlim[1]*split/2,max(y_train)*0.9))
plt.show()
