import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import tensorflow as tf
from make_data import *
# tf.config.experimental_run_functions_eagerly()

tfd = tfp.distributions

xlim=[0, 1]
ylim=[0, 1]
split = 0.25
length = 100
batch_size = 32
epochs = 1000

# get data
x_train, y_train, x_test, y_test, y_stddev, y_mean = get_data(length=length, width=1, mean=0, stddev=0.25, split=split,
                                                      xlim=xlim, ylim=ylim)

# build model
negloglik = lambda y, p_y: -p_y.log_prob(y)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, ),
    tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1],
                             scale=1e-5+tf.math.abs(t[..., 1:]))),
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=negloglik)

model.fit(x_train, y_train,
          epochs=epochs, batch_size=batch_size)

# predict train
yhat = model(x_train)
mean = yhat.mean()
stddev = yhat.stddev()

interval_dict = {50:0.67, 80:1.282,85:1.440, 90:1.65, 95:1.96, 99:2.58, 99.5:2.807,99.9:3.291}
conf = 90

# plot train
plt.scatter(x_train, y_train, marker='+', color='black')
plt.plot(x_train, mean, color='red', linewidth=2)

plt.fill_between(x_train[:,0], mean[:,0] + interval_dict[conf] * stddev[:,0], mean[:,0] - interval_dict[conf] * stddev[:,0], color = 'pink', alpha = 0.5)

# predict test
yhat = model(x_test)
mean = yhat.mean()
stddev = yhat.stddev()


# plot test
plt.scatter(x_test, y_test,marker='+',color='black', label = 'training data')
plt.fill_between(x_test[:,0], mean[:,0] + interval_dict[conf] * stddev[:,0], mean[:,0] - interval_dict[conf] * stddev[:,0], color = 'pink', alpha = 0.5, label = str(conf)+'% confidence')
plt.plot(x_test, mean, color='red', linewidth=2, label = 'mean prediction')
plt.xlabel('Input')
plt.ylabel('Output')

plt.legend()
# tidy up graph
# plt.plot([xlim[1]*(1-split), xlim[1]*(1-split)], [-100, 100], color = 'black')
plt.ylim((min(y_train), max(y_train)))
# plt.annotate('train', (xlim[1]*(1-split)/2,max(y_train)*0.9))
# plt.annotate('test', (xlim[1] - xlim[1]*split/2,max(y_train)*0.9))
plt.title('Aleatoric Uncertainty')
plt.savefig('aleatoric uncertainty.png')
plt.show()

