import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp
import tqdm
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
tfd = tfp.distributions
tf.random.set_seed(0)


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
            for layer in self.model.layers:
                KL = KL + kl_loss_weight*tfp.distributions.kl_divergence(layer._posterior(np.ones(layer.input_shape[1])),
                                                layer._prior(np.ones(layer.input_shape[1]))).numpy()


            self.loss.append(self.model.history.history['loss'])
            self.lik_loss.append(self.model.history.history['loss'])
            self.kl_loss.append(KL)
            # print('KL = ', KL, 'Likelihood = ', lik_loss)

        return model

    def plot(self, axes = 1):
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('epoch')

        ax1.plot(self.lik_loss, label = 'likelihood', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylabel('likelihood', color=color)
        # ax1.set_ylim((0,100))

        if axes == 2:
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

            color = 'tab:blue'
            ax2.set_ylabel('KL divergence', color=color)  # we already handled the x-label with ax1
            ax2.plot(self.kl_loss, label = 'KL divergence', color=color)
            ax2.tick_params(axis='y', labelcolor=color)

        else:
            # ax1.plot(self.kl_loss, label='KL divergence', color=color)
            ax1.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

        plt.legend()
        plt.show()

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
	dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
	return K.sum(-dist.log_prob(y_true))

# Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(0.4))
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
            tfd.Normal(loc=t, scale=0.4),
            reinterpreted_batch_ndims=1)),
    ])

def f(x, sigma, scale):
	epsilon = np.random.randn(*x.shape) * sigma
	return  scale * np.sin(2 * np.pi * (x)) + epsilon

train_size = 8
noise = 0.01
scale = 1.0

X = np.linspace(-0.5, 0.5, train_size).reshape(-1, 1)

k = [0]
for i in(np.random.rand(train_size-1)):
    k.append(k[-1]+i)
k = k/max(k) - 0.5
X = np.asarray(k)

y = f(X, sigma=noise, scale = scale)
y_true = f(X, sigma=0.0, scale = scale)

# plt.scatter(X, y, marker='+', label='Training data')
# plt.plot(X, y_true, label='Truth')
# plt.title('Noisy training data and ground truth')
# plt.legend()
# plt.show()

batch_size = train_size
num_batches = train_size / batch_size
kl_loss_weight = 1.0 / num_batches

# Build model.

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tfp.layers.DenseVariational(units=64,
                                make_posterior_fn=posterior_mean_field,
                                make_prior_fn=prior_trainable,
                                kl_weight=kl_loss_weight,
                                activation='relu'),
    tfp.layers.DenseVariational(units=64,
                                make_posterior_fn=posterior_mean_field,
                                make_prior_fn=prior_trainable,
                                kl_weight=kl_loss_weight,
                                activation='relu'),
    tfp.layers.DenseVariational(units=1,
                                make_posterior_fn=posterior_mean_field,
                                make_prior_fn=prior_trainable,
                                kl_weight=kl_loss_weight)
])

model.compile(loss=neg_log_likelihood, optimizer=Adam(lr=0.03), metrics=['mse'])

trainer = Train(model, 2000, 32)


model = trainer.fit(X, y)
# model.fit(X, y, batch_size=batch_size, epochs=1500, verbose=0)
trainer.plot1()


X_test = np.linspace(-1.5, 1.5, 100).reshape(-1, 1)
y_pred_list = []
for i in tqdm.tqdm(range(500)):
    y_pred = model.predict(X_test)
    y_pred_list.append(y_pred)

y_preds = np.concatenate(y_pred_list, axis=1)
y_mean = np.mean(y_preds, axis=1)
y_sigma = np.std(y_preds, axis=1)

plt.plot(X_test, y_mean, 'r-', label='Predicted mean');
plt.scatter(X, y, marker='+', label='Training data')
plt.fill_between(X_test.ravel(),
                 y_mean + 2 * y_sigma,
                 y_mean - 2 * y_sigma,
                 alpha=0.5, label='Predicted uncertainty')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Prediction')
plt.legend()
plt.show()



def mixture_prior_params(sigma_1, sigma_2, pi, return_sigma=False):
    params = K.variable([sigma_1, sigma_2, pi], name='mixture_prior_params')
    sigma = np.sqrt(pi * sigma_1 ** 2 + (1 - pi) * sigma_2 ** 2)
    return params, sigma


def log_mixture_prior_prob(w):
    comp_1_dist = tf.distributions.Normal(0.0, prior_params[0])
    comp_2_dist = tf.distributions.Normal(0.0, prior_params[1])
    comp_1_weight = prior_params[2]
    return K.log(comp_1_weight * comp_1_dist.prob(w) + (1 - comp_1_weight) * comp_2_dist.prob(w))

