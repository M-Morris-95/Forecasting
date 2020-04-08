import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp
import tqdm
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
tfd = tfp.distributions


class Train:
    def __init__(self, model, epochs, batch_size):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.lik_loss = []
        self.kl_loss = []
        self.loss = []

    def fit(self, x_train, y_train):

        for i in tqdm.tqdm(range(self.epochs)):
            model.fit(x_train, y_train,
                      epochs=1,
                      batch_size=self.batch_size,
                      verbose = 0)

            KL = 0
            for layer in self.model.layers[:-1]:
                KL = KL + kl_loss_weight*tfp.distributions.kl_divergence(layer._posterior(np.ones(layer.input_shape[1])),
                                                layer._prior(np.ones(layer.input_shape[1]))).numpy()


            self.loss.append(self.model.history.history['loss'])
            self.lik_loss.append(self.model.history.history['loss'])
            self.kl_loss.append(KL)
            # print('KL = ', KL, 'Likelihood = ', lik_loss)

        return model

    def plot1(self):
        self.color1 = 'tab:red'
        self.color2 = 'tab:blue'
        plt.plot(self.lik_loss, label='likelihood loss', color=self.color1)
        plt.plot(self.kl_loss, label='KL loss', color=self.color2)
        plt.legend
        plt.ylim((-100,200))
        plt.legend()
        plt.show()

def neg_log_likelihood(y_true, y_pred, sigma=0.1):
	return K.sum(-y_pred.log_prob(y_true))

# Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(0.2))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                       scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])


# Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t, scale=0.2),
            reinterpreted_batch_ndims=1)),
    ])

def f(x, sigma, scale):
	epsilon = np.random.randn(*x.shape) * sigma
	return  scale * np.sin(2 * np.pi * (x)) + epsilon

train_size = 128
noise = 0.35
scale = 1.0

X = np.linspace(-0.5, 0.5, train_size).reshape(-1, 1)
y = f(X, sigma=noise, scale = scale)
y_true = f(X, sigma=0.0, scale = scale)


plt.scatter(X, y, marker='+', label='Training data')
plt.plot(X, y_true, label='Truth')
plt.title('Noisy training data and ground truth')
plt.legend()
plt.ylim((-2,2))
plt.show()



# Mixture prior parameters shared across DenseVariational layer instances
# prior_params, prior_sigma = mixture_prior_params(sigma_1=1.0, sigma_2=0.1, pi=0.2)

batch_size = train_size
num_batches = train_size / batch_size
kl_loss_weight = 1.0 / num_batches





# Build model.
c = np.log(np.expm1(0.1))
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tfp.layers.DenseVariational(units=20,
                                make_posterior_fn=posterior_mean_field,
                                make_prior_fn=prior_trainable,
                                kl_weight=kl_loss_weight,
                                activation='relu'),
    tfp.layers.DenseVariational(units=20,
                                make_posterior_fn=posterior_mean_field,
                                make_prior_fn=prior_trainable,
                                kl_weight=kl_loss_weight,
                                activation='relu'),
    tfp.layers.DenseVariational(units=2,
                                make_posterior_fn=posterior_mean_field,
                                make_prior_fn=prior_trainable,
                                kl_weight=kl_loss_weight),
tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1],
                           scale=1e-3 + tf.math.softplus(-2.2521684610440906+t[..., 1:]))),
])




model.compile(loss=neg_log_likelihood, optimizer=Adam(lr=0.03), metrics=['mse'])

trainer = Train(model, 2500, 32)


model = trainer.fit(X, y)
# model.fit(X, y, batch_size=batch_size, epochs=1500, verbose=0)
trainer.plot1()


X_test = np.linspace(-1.5, 1.5, 100).reshape(-1, 1)
y_pred_list = []
y_pred_means = []
y_pred_stds = []
for i in tqdm.tqdm(range(500)):
    y_pred = model(X_test)
    y_pred_means.append(y_pred.mean())
    y_pred_stds.append(y_pred.stddev())

y_means = np.concatenate(y_pred_means, axis=1)
mean_mean = np.mean(y_means, axis=1)
mean_sigma = np.std(y_means, axis=1)

y_stds = np.concatenate(y_pred_stds, axis=1)
std_mean = np.mean(y_stds, axis=1)
std_sigma = np.std(y_stds, axis=1)
plt.fill_between(X_test.ravel(),
                 mean_mean + mean_sigma,
                 mean_mean - mean_sigma,
                 alpha=0.3, label='Epistemic uncertainty')
plt.fill_between(X_test.ravel(),
                 mean_mean + std_mean,
                 mean_mean - std_mean,
                 alpha=0.3, label='Aleatoric uncertainty')
plt.plot(X_test, mean_mean, 'r-', label='Predictive mean');
plt.scatter(X, y, marker='+', label='Training data')
plt.title('Prediction')
plt.legend()
plt.ylim((-2,2))
plt.show()


