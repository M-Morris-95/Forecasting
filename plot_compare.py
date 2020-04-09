import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.dates import DateFormatter
import datetime as dt


logging_root = '/Users/michael/Documents/github/Forecasting/' + 'Logging'

comp1 = '/LINEAR_14LA_Apr_06_10_09'
comp2 = '/GRU_COMBINED_UNCERTAINTY_14LA_Apr_07_10_42'

pred_name = '/test_predictions.csv'
truth_name = '/test_ground_truth.csv'

simons_file = '/Users/michael/Documents/github/data_forecasting_ILI/forecasts/days_ahead_14/ensembles/eng_GRU_MI_SO_ensembles_forecasts'
simons_forecasts = pd.read_csv(simons_file, header=None)

idx = []
for i in range(1461):
    if (simons_forecasts[0][i][-5:] == '08-24'):
        idx.append(i)

temp = np.asarray(simons_forecasts[1])


forecasts = np.empty((4,366))
forecasts[:] = np.nan

forecasts[0, :365] = temp[:idx[1]]
forecasts[1, :] = temp[idx[1]:idx[2]]
forecasts[2, :365] = temp[idx[2]:idx[3]]
forecasts[3, :365] = temp[idx[3]:]


linear = pd.read_csv(logging_root+comp1+pred_name)

yeah_end = '08-24'

mean = pd.read_csv(logging_root+comp2+pred_name)
std = pd.read_csv(logging_root+comp2+'/test_stddev.csv')
truth = pd.read_csv(logging_root+comp2+truth_name)

plt.figure(1, figsize = [9,7])
formatter = DateFormatter('%b')
for i in range(1,5):
    plt.subplot(2,2,i)

    x_ax = []
    for j in np.linspace(0, 365, 366):
        x_ax.append(dt.datetime.strptime(str(2013+i) + ' 8 23', '%Y %m %d') + dt.timedelta(days=j))

    plt.plot(x_ax, forecasts[i-1, :], '-.', color = 'blue', label = 'GRU-MI-MO')
    plt.plot(x_ax, truth.iloc[:,i], '-', color = 'black', label = 'ground truth')
    # plt.plot(x_ax, linear.iloc[:, i], '-.', color = 'black', label = 'linear model')
    plt.plot(x_ax, mean.iloc[:, i], '--',color = 'red', label = 'uncertainty mean')
    plt.fill_between(x_ax,  mean.iloc[:, i] + std.iloc[:, i], mean.iloc[:, i] - std.iloc[:, i], color = 'pink', label = 'confidence', alpha = 0.5)

    plt.gcf().axes[i-1].xaxis.set_major_formatter(formatter)
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.title(str(2013+i) + '/' + str(14+i), fontsize = 8)
    plt.ylabel('ILI rate', fontsize = 8)
    plt.legend(fontsize = 8)

plt.savefig('comparison.png')
plt.show()