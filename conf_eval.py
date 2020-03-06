import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

root = '/Users/michael/Documents/Confidence_Eval/'
y_pred = np.asarray(pd.read_csv(root + 'test_predictions.csv', header=0, index_col = 0))
y_std = 3*np.asarray(pd.read_csv(root + 'test_stddev.csv', header=0, index_col = 0))
y_true = np.asarray(pd.read_csv(root + 'test_ground_truth.csv', header=0, index_col = 0))

plt.figure(1)
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(y_true[:, i], color="blue", alpha=1, label="ground_truth")
    plt.plot(y_pred[:, i], color="red", alpha=1, label="ground_truth")
    plt.fill_between(np.linspace(1, y_true.shape[0], y_true.shape[0]), np.squeeze(y_pred[:, i] - y_std[:, i]),
                     np.squeeze(y_pred[:, i] + y_std[:, i]),
                     color="pink", alpha=0.5, label="predict std")

plt.show()
