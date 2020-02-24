import numpy as np
import tensorflow as tf
import metrics
from Parser import GetParser
from Functions import data_builder,logger, plotter
import matplotlib.pyplot as plt
import pandas as pd
from Bayesian_Functions import *
import tensorflow_probability as tfp
from sklearn.linear_model import BayesianRidge,  LinearRegression
parser = GetParser()
args = parser.parse_args()

EPOCHS, BATCH_SIZE = args.Epochs, args.Batch_Size

logging = logger(args)
models, look_aheads, max_k = logging.get_inputs()
look_ahead = 14
tf.random.set_seed(0)





for fold_num in range(1,2):
    logging.update_details(fold_num=fold_num, k=1, model='Bayes', look_ahead=look_ahead)

    data = data_builder(args, fold=fold_num, look_ahead=look_ahead)
    x_train, y_train, y_train_index, x_test, y_test, y_test_index = data.build()

    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))
    y_train = y_train[:, -1]
    y_test = y_test[:, -1]
    br = BayesianRidge(fit_intercept=False, tol=1e-5, verbose=True)
    # lm = LinearRegression(fit_intercept=False)
    br.fit(x_train, y_train)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    y_pred, ystd = br.predict(x_test, return_std=True)

    # lm.fit(x_train,y_train)

    plt.plot(np.linspace(1, x_test.shape[0], x_test.shape[0]),y_test, color="blue", label="sin($2\\pi x$)")
    # plt.scatter(x_train, y_train, s=50, alpha=0.5, label="observation")
    plt.plot(np.linspace(1, x_test.shape[0], x_test.shape[0]),y_pred, color="red", label="predict mean")
    plt.fill_between(np.linspace(1, x_test.shape[0], x_test.shape[0]),y_pred - ystd, y_pred + ystd,
                    color="pink", alpha=0.5, label="predict std")
    # plt.plot(np.linspace(1, x_test.shape[0], x_test.shape[0]), lm.predict(x_test), color="green")
    plt.show()


#
# N_list = [3, 8, 50]
#
# beta = 25.0
# alpha = 2.0
#
# # Training observations in [-1, 1)
# X = np.random.rand(N_list[-1], 1)
#
# # Training target values
# t = g(X, noise_variance=1 / beta)
#
# # Test observations
# X_test = np.linspace(0, 1, 100).reshape(-1, 1)
#
# # Function values without noise
# y_true = g(X_test, noise_variance=0)
#
# # Design matrix of test observations
# Phi_test = expand(X_test, bf=gaussian_basis_function, bf_args=np.linspace(0, 1, 9))
#
# plt.figure(figsize=(10, 10))
# plt.subplots_adjust(hspace=0.4)

# for i, N in enumerate(N_list):
#     X_N = X[:N]
#     t_N = t[:N]
#
#     # Design matrix of training observations
#     Phi_N = expand(X_N, bf=gaussian_basis_function, bf_args=np.linspace(0, 1, 9))
#
#     # Mean and covariance matrix of posterior
#     m_N, S_N = posterior(Phi_N, t_N, alpha, beta)
#
#     # Mean and variances of posterior predictive
#     y, y_var = posterior_predictive(Phi_test, m_N, S_N, beta)
#
#     # Draw 5 random weight samples from posterior and compute y values
#     w_samples = np.random.multivariate_normal(m_N.ravel(), S_N, 5).T
#     y_samples = Phi_test.dot(w_samples)
#
#     plt.subplot(len(N_list), 2, i * 2 + 1)
#     plot_data(X_N, t_N)
#     plot_truth(X_test, y_true)
#     plot_posterior_samples(X_test, y_samples)
#     plt.ylim(-1.0, 2.0)
#     plt.legend()
#
#     plt.subplot(len(N_list), 2, i * 2 + 2)
#     plot_data(X_N, t_N)
#     plot_truth(X_test, y_true, label=None)
#     plot_predictive(X_test, y, np.sqrt(y_var))
#     plt.ylim(-1.0, 2.0)
#     plt.legend()
# plt.show()