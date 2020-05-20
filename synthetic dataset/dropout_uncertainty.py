import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def f(x, sigma, scale):
	epsilon = np.random.randn(*x.shape) * sigma
	return  scale * np.sin(2 * np.pi * (x)) + epsilon

train_size = 256
noise = 0.35
scale = 1.0

X = np.linspace(-0.5, 0.5, train_size).reshape(-1, 1)
y = f(X, sigma=noise, scale = scale)
y_true = f(X, sigma=0.0, scale = scale)




dropout_amount = 0.25
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(dropout_amount),
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(dropout_amount),
    tf.keras.layers.Dense(1,  activation=tf.keras.activations.linear),
])

model.compile(loss=tf.keras.losses.mean_squared_error,
              optimizer=tf.keras.optimizers.Adam(lr=0.03),
              )

model.fit(X, y,
          epochs = 100,
          batch_size=16
)

model.predict(X)

predictions = []
for i in range(500):
    predictions.append(model(X, training=True))

predictions = np.asarray(predictions)
pred_mean = np.mean(predictions, 0)
pred_std = np.std(predictions, 0)

plt.scatter(X, y, marker='+', label='Training data')
plt.plot(X, y_true, 'r-', label='Truth')

plt.fill_between(np.squeeze(X),
                 np.squeeze(pred_mean-pred_std),
                 np.squeeze(pred_mean+pred_std),
                 alpha = 0.5,
                 color = 'pink',
                 label='uncertainty')

plt.plot(X, model.predict(X), label='Prediction')
plt.title('Noisy training data and ground truth')
plt.legend()
plt.ylim((-2,2))
plt.show()




X_test = np.linspace(-1.5, 1.5, 100).reshape(-1, 1)

predictions = []
for i in range(500):
    predictions.append(model(X_test, training=True))

X_test = np.squeeze(X_test)
predictions = np.squeeze(np.asarray(predictions))
pred_mean = np.mean(predictions, 0)
pred_std = np.std(predictions, 0)


plt.scatter(X, y, marker='+', label='Training data')

plt.plot(X_test, pred_mean, 'r-', label='Predicted mean')
plt.fill_between(np.squeeze(X_test),
                 np.squeeze(pred_mean-pred_std),
                 np.squeeze(pred_mean+pred_std),
                 alpha = 0.5,
                 color = 'pink',
                 label='uncertainty')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Prediction on test set')
plt.legend()
plt.show()

