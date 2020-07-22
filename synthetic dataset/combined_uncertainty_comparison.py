import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
                try:
                    KL = KL + kl_loss_weight*tfp.distributions.kl_divergence(layer._posterior(np.ones(layer.input_shape[1])),
                                                layer._prior(np.ones(layer.input_shape[1]))).numpy()
                except:
                    pass

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
# trainer = Train(model, epochs=200, batch_size=batch_size)
# model = trainer.fit(X, y)

def neg_log_likelihood(y_true, y_pred, sigma=0.1):
	dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
	return K.sum(-dist.log_prob(y_true))

# Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size

    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                       scale=1e-5 + 0.1*tf.nn.softplus(10*t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])

# Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t, scale=5),
            reinterpreted_batch_ndims=1)),
    ])



data = pd.read_csv('training_data.csv')
X = data.x_train.values
y = data.y_train.values

train_size = 256
batch_size = 16
epochs = 2000
scale = 1
minmax = 1
KL_anneal = 0.1
X = (np.linspace(-minmax, minmax, train_size))[:, np.newaxis]
y = np.sin(np.pi*X/minmax)+np.random.rand(train_size)[:, np.newaxis]*scale*X/minmax - (scale*X/minmax)/2



kl_loss_weight = KL_anneal * batch_size / train_size

# Build model.
c = np.log(np.expm1(0.1))
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(64, activation = 'relu'),
    # tf.keras.layers.Dense(2, activation = 'linear'),
    # tfp.layers.DenseVariational(units=64,
    #                             make_posterior_fn=posterior_mean_field,
    #                             make_prior_fn=prior_trainable,
    #                             kl_weight=kl_loss_weight,
    #                             activation='relu'),
    tfp.layers.DenseVariational(units=2,
                                make_posterior_fn=posterior_mean_field,
                                make_prior_fn=prior_trainable,
                                kl_weight=kl_loss_weight),
    tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1],
                             # scale = 0.1))
                             scale=1e-5+tf.math.softplus(t[..., 1:]))),
])





negloglik = lambda y, p_y: -p_y.log_prob(y)

model.compile(loss=negloglik, optimizer=Adam(), metrics=['mse'])
likelihood = []
KL = []

epoch_loss = 2
for i in tqdm.tqdm(range(int(epochs/epoch_loss))):
    model.fit(X, y,
              epochs = epoch_loss,
              batch_size = batch_size,
              verbose = False)

    activation = X[:, np.newaxis].copy()
    for layer in model.layers:
        try:
            prior = layer._prior(activation)
            posterior = layer._posterior(activation)

        except:
            pass
        activation = layer(activation)

    likelihood.append(np.mean(negloglik(y, activation).numpy()))
    KL.append(KL_anneal*tfp.distributions.kl_divergence(posterior, prior).numpy())




try:
    plt.plot(likelihood, label = 'likelihood')
    plt.plot(KL, label = 'KL')
    plt.legend()
    plt.show()
except:
    pass

cont = True
if cont:

    X_test = np.linspace(-1.5, 1.5, 100).reshape(-1, 1)

    predictions = []
    means = []

    for i in range(25):
        p = model(X_test)
        # predictions.append(p.mean())
        for j in range(25):
            predictions.append(p.sample())

    pred_mean = np.mean(np.squeeze(np.asarray(predictions)), 0)
    pred_std = np.std(np.squeeze(np.asarray(predictions)), 0)

    interval_dict = {50:0.67, 80:1.282,85:1.440, 90:1.65, 95:1.96, 99:2.58, 99.5:2.807,99.9:3.291}
    conf = 90

    plt.figure(1)
    plt.scatter(X, y, marker='+',color='black', label='Training data')

    plt.plot( np.squeeze(X_test), pred_mean, 'r-', label='Predicted mean')
    plt.fill_between( np.squeeze(X_test).ravel(),
                     pred_mean + interval_dict[conf] * pred_std,
                     pred_mean - interval_dict[conf] * pred_std,
                     color='pink',
                     alpha=0.5,
                     label='Predicted uncertainty')

    # plt.plot([-minmax,-minmax], [-2, 2],
    #          color = 'black',
    #          linestyle = 'dashed',
    #          linewidth = 2)
    # plt.plot([minmax, minmax], [-2, 2],
    #          color = 'black',
    #          linestyle='dashed',
    #          linewidth = 2)
    plt.ylim([-2,2])
    plt.xlim([-1.5,1.5])
    # plt.text(-1, -1.8, 'out of sample', fontsize=10, horizontalalignment='center')
    # plt.text( 1, -1.8, 'out of sample', fontsize=10, horizontalalignment='center')
    # plt.text( 0, -1.8, 'training set' , fontsize=10, horizontalalignment='center')
    plt.xlabel('Input' , fontsize = 10)
    plt.ylabel('Output', fontsize = 10)
    plt.legend(fontsize=8, loc='upper center')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.title('Combined Aleatoric and Epistemic Uncertainty')
    # plt.savefig('combined uncertainty.png')
    plt.show()