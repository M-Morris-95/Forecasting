import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

root = '/Users/michael/Documents/github/Forecasting/Logging/'
folder = 'GRU_Search_14LA_Mar_23_10_27/'
pred_file = 'test_predictions.csv'
true_file = 'test_ground_truth.csv'
stdd_file ='test_stddev.csv'

y_pred = pd.read_csv(root+folder+pred_file).drop('Unnamed: 0', 1).values
y_true = pd.read_csv(root+folder+true_file).drop('Unnamed: 0', 1).values
stddev = pd.read_csv(root+folder+stdd_file).drop('Unnamed: 0', 1).values

MAE = np.mean(np.abs(y_true-y_pred), 1)
# np.abs(y_true-y_pred) >= stddev

for i in range(1):
    plt.subplot(2,2,i+1)
    plt.plot(y_pred[:,i])
    plt.plot(y_true[:,i])
    plt.plot(y_pred[:,i]-stddev[:,i])
    plt.plot(y_pred[:,i]+stddev[:,i])

plt.show()