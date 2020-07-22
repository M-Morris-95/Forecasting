import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import datetime as dt
import os

logging_root = '/Users/michael/Documents/github/Forecasting/Logging'



pred_name = '/test_predictions.csv'
truth_name = '/test_ground_truth.csv'

def rmse(error):
    return np.sqrt(np.mean(np.square(error)))

def mse(error):
    return np.mean(np.square(error))

def mae(error):
    return np.mean(np.abs(error))

year_end = '08-24'

log = '/GRU_DATA_UNCERTAINTY_14LA_Jun_17_12_38'
log = '/GRU_DATA_UNCERTAINTY_14LA_Jun_17_13_22'
mean = pd.read_csv(logging_root+log+pred_name)
std = pd.read_csv(logging_root+log+'/test_stddev.csv')
truth = pd.read_csv(logging_root+log+truth_name)

plt.figure(1, figsize = [9,7], dpi=500)
formatter = DateFormatter('%b')

errors = pd.DataFrame(index=['mae','mse','rmse', 'Corr'])

interval_dict = {50:0.67, 80:1.282,85:1.440, 90:1.65, 95:1.96, 99:2.58, 99.5:2.807,99.9:3.291}

intervals = [50, 90, 85, 90, 95, 99, 99.5, 99.9]
intervals = [95]
ylim = [[0,30],
        [0,30],
        [0,30],
        [0,60]]


for i in range(1,5):
    plt.subplot(2,2,i)

    x_ax = []
    for j in np.linspace(0, 365, 366):
        x_ax.append(dt.datetime.strptime(str(2013+i) + ' 8 23', '%Y %m %d') + dt.timedelta(days=j))

    plt.plot(x_ax, truth.iloc[:,i],
             '-',
             color = 'black',
             label = 'ground truth')
    plt.plot(x_ax, mean.iloc[:, i],
             '--',
             color = 'red',
             label = 'GRU Epistemic Uncertainty')

    for interval in intervals:
        plt.fill_between(x_ax,
                         mean.iloc[:, i] + interval_dict[interval]*std.iloc[:, i],
                         mean.iloc[:, i] - interval_dict[interval]*std.iloc[:, i],
                         color = 'red',
                         linewidth=0.0,
                         # label = 'model uncertainty confidence interval',
                         alpha = 0.2)

    plt.gcf().axes[i-1].xaxis.set_major_formatter(formatter)
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.title(str(2013+i) + '/' + str(14+i), fontsize = 8)
    plt.ylabel('ILI rate', fontsize = 8)
    plt.legend(fontsize = 8)
    plt.ylim(ylim[i-1])
    my_error = truth.iloc[:,i] - mean.iloc[:, i]

    errors['GRU-VI-' + str(2013 + i) + '/' + str(14 + i)] = [mae(my_error), mse(my_error),
                                                                rmse(my_error), np.corrcoef(truth.iloc[:-1,i] , mean.iloc[:-1, i])[0,1]]

os.chdir('../')
plt.savefig('GRU_Model_Uncertainty 14LA.png')
plt.show()