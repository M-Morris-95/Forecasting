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
np.random.seed(seed=0)


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

epochs = [800,100]
for iter, train_size in enumerate([8, 64]):
    # train_size = 8
    noise = 0.1
    scale = 1.0

    X = np.linspace(-0.5, 0.5, train_size).reshape(-1, 1)

    k = [0]
    for i in(np.random.rand(train_size-1)):
        k.append(k[-1]+i)
    k = k/max(k) - 0.5
    X = np.asarray(k)

    y = f(X, sigma=noise, scale = scale)
    y_true = f(X, sigma=0.0, scale = scale)

    batch_size = 4
    kl_anneal = 2
    num_batches = train_size / batch_size
    kl_loss_weight = kl_anneal / num_batches

    # Build model.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),
        tfp.layers.DenseVariational(units=32,
                                    make_posterior_fn=posterior_mean_field,
                                    make_prior_fn=prior_trainable,
                                    kl_weight=kl_loss_weight,
                                    activation='relu'),
        # tfp.layers.DenseVariational(units=8,
        #                             make_posterior_fn=posterior_mean_field,
        #                             make_prior_fn=prior_trainable,
        #                             kl_weight=kl_loss_weight,
        #                             activation='relu'),
        tfp.layers.DenseVariational(units=1,
                                    make_posterior_fn=posterior_mean_field,
                                    make_prior_fn=prior_trainable,
                                    kl_weight=kl_loss_weight)
    ])

    model.compile(loss=neg_log_likelihood, optimizer=Adam(lr=0.03), metrics=['mse'])

    trainer = Train(model, epochs=epochs[iter], batch_size=batch_size)

    model = trainer.fit(X, y)
    # trainer.plot1()



    # predictions = []
    # for i in range(100):
    #     predictions.append(model.predict(X))
    #
    # predictions = np.squeeze(np.asarray(predictions))
    # pred_mean = np.mean(predictions, 0)
    # pred_std = np.std(predictions, 0)
    #
    # plt.scatter(X, y, marker='+', label='Training data')
    #
    # plt.plot(X, pred_mean, 'r-', label='Predicted mean')
    # plt.fill_between(X.ravel(),
    #                  pred_mean + 2 * pred_std,
    #                  pred_mean - 2 * pred_std,
    #                  color='pink',
    #                  alpha=0.5,
    #                  label='Predicted uncertainty')
    #
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.title('Prediction on Training Set')
    # plt.legend()
    # plt.show()






    X_test = np.linspace(-1.5, 1.5, 100).reshape(-1, 1)

    predictions = []
    for i in range(100):
        predictions.append(model.predict(X_test))

    X_test = np.squeeze(X_test)
    predictions = np.squeeze(np.asarray(predictions))
    pred_mean = np.mean(predictions, 0)
    pred_std = np.std(predictions, 0)
    legends = ['prediction 8 train points','prediction 64 train points']
    color = ['blue', 'red', 'cornflowerblue','pink']

    if iter == 1:
        plt.scatter(X, y,
                    marker='+',
                    color = 'black',
                    label = 'training data')
    plt.plot(X_test, pred_mean,
             color=color[iter],
             label=legends[iter])
    plt.fill_between(X_test.ravel(),
                     pred_mean + 2 * pred_std,
                     pred_mean - 2 * pred_std,
                     color=color[iter+2],
                     alpha=0.5)
    plt.ylim([-2,2])

    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title('Epistemic Uncertainty')
    plt.legend()
plt.savefig('aleatoric complex.png')
plt.show()