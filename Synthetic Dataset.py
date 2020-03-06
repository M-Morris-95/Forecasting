import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
import matplotlib.pyplot as plt
from metrics import *
length = 10028
width = 1
mean = 0
split = 0.25
stddev = 0.2

EPOCHS = 3

BATCH_SIZE=128
tf.random.set_seed(1)
x_train = np.zeros((length,width))
y_train = np.zeros((length, 1))
for j in range(x_train.shape[1]):
    for i in range(x_train.shape[0]):
        x_train[i, j] = (i +  (np.random.normal(mean, 0.1)))

for i in range(y_train.shape[0]):
    N = 4
    temp = 2*(x_train[i, 0]/length)**1- 6*(x_train[i, 0]/length)**2 + 2*(x_train[i, 0]/length)**3 + 2*(x_train[i, 0]/length)**4
    temp = 4 * (x_train[i, 0] / length)


    # temp = 5*(np.sin(4*N*np.pi*(x_train[i, 1])/length)+np.cos(2*N*np.pi*(x_train[i, 1])/length))

    y_train[i] = temp + 0.5*temp*(np.random.normal(mean, stddev))


''' DATASET 
5 SIN(2 PI X/5000) + N(MEAN=0, STD=0.2)
'''

x_data = []
y_data = []
lag = 28
for i in range(length - lag-1):
    x_data.append(x_train[i:i+lag])
    y_data.append(y_train[i+lag+1])

x_train=np.asarray(x_data)
y_train=np.asarray(y_data)

x_train = x_train/np.max(x_train)
y_train = y_train - np.min(y_train)
# y_train = y_train/np.max(y_train)

split_te = int((split)*(length-lag))
split_tr = int((1-split)*(length-lag))

x_test = x_train[-split_te:]
y_test = y_train[-split_te:]

x_train= x_train[:-split_te]
y_train = y_train[:-split_te]

plt.scatter(x_train[:, 0, 0], y_train, s=0.2, alpha=0.5)
plt.scatter(x_test[:, 0, 0], y_test, s=0.2, alpha=0.5)
plt.show()




def normal_scale_uncertainty(t, softplus_scale=0.2):
    """Create distribution with variable mean and variance"""
    return tfd.Normal(loc=t[..., :1],
                      scale=1e-3*tf.math.softplus(softplus_scale * t[..., 1:]))

initializer = tf.keras.initializers.glorot_normal(seed=None)
confidence = True
tfd = tfp.distributions

loss = lambda y, p_y: -p_y.log_prob(y)

ili_input = tf.keras.layers.Input(shape=[x_train.shape[1], x_train.shape[2]])
x = tf.keras.layers.LSTM(x_train.shape[1], activation='relu', return_sequences=True)(
    ili_input)
# x = tf.keras.layers.LSTM(int((x_train.shape[2] - 1)), activation='relu', return_sequences=True,kernel_initializer=initializer)(x)
y = tf.keras.layers.LSTM(10, activation='relu', return_sequences=False,kernel_initializer=initializer)(x)
z = tf.keras.layers.Dense(2, activation='relu',kernel_initializer=initializer)(y)
z = tfp.layers.DistributionLambda(normal_scale_uncertainty)(z)
model = tf.keras.models.Model(inputs=ili_input, outputs=z)

optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['mae', 'mse', rmse])

model.fit(x_train, y_train,
                validation_data = (x_test,y_test),
                epochs=EPOCHS, batch_size=BATCH_SIZE)

# model2.fit(x_train, y_train,
#                 validation_data = (x_test,y_test),
#                 epochs=EPOCHS, batch_size=BATCH_SIZE)


yhat = model(x_train)
y_pred = yhat.mean()
y_std = yhat.stddev()
plt.plot(x_train[:, 0, 0], y_pred, color="lime", label="train prediction")
plt.scatter(x_train[:, 0, 0], y_train, s=0.2, alpha=0.5, color="mediumslateblue", label="train ground_truth")
plt.fill_between(x_train[:, 0, 0], np.squeeze(y_pred - y_std), np.squeeze(y_pred + y_std),
                 color="aquamarine", alpha=0.5, label="train predict std")

yhat = model(x_test)
y_pred = yhat.mean()
y_std = yhat.stddev()

plt.plot(x_test[:, 0, 0], y_pred, color="red", label="prediction")
plt.scatter(x_test[:, 0, 0], y_test, s=0.2, alpha=0.5, color="blue", label="ground_truth")
plt.fill_between(x_test[:, 0, 0], np.squeeze(y_pred - y_std), np.squeeze(y_pred + y_std),
                 color="pink", alpha=0.5, label="predict std")
# plt.plot(x_test[:, -1, 0], model2.predict(x_test),  label="no conf")

plt.legend(fontsize=8)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

plt.show()

plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.show()

