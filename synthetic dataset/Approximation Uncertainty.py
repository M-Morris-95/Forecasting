import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def f(x, sigma, scale):
	epsilon = np.random.randn(*x.shape) * sigma
	return  scale * np.sin(2 * np.pi * (x)) + epsilon

X = np.linspace(-0.5, 0.5, 50).reshape(-1, 1)

y = f(X, sigma=0, scale = 1)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(2, activation = 'relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss = 'mse')

model.fit(X,y, epochs = 500)

plt.scatter(X,y, marker='+', color = 'black', label = 'training data')

plt.plot(X,model.predict(X), color = 'blue', label = 'prediction simple')

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(10, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss = 'mse')

model.fit(X,y, epochs = 800)

plt.plot(X,model.predict(X), color = 'red', label = 'prediction complex')



plt.legend()
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Approximation Uncertainty')
plt.savefig('approximation uncertainty complex.png')
plt.show()