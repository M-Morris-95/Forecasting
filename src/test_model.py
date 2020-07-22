import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
from matplotlib.dates import DateFormatter
formatter = DateFormatter('%b')

covid = pd.read_csv('/Users/michael/Documents/datasets/covid_deaths.csv')
dates = []
for i in range(len(covid)-1):
    dates.append(dt.datetime.strptime(covid['NHS England Region'][i], '%d-%b-%y').date())
covid = pd.DataFrame(data={'date': np.asarray(dates), 'covid':np.asarray(covid['England'][:-1])})

for fold in range(4,5):
    in_sample = pd.read_csv('/Users/michael/Documents/datasets/in_sample'+str(fold)+'.csv')
    out_of_sample = pd.read_csv('/Users/michael/Documents/datasets/out_of_sample'+str(fold)+'.csv')


    interval_dict = {50: 0.67, 80: 1.282, 85: 1.440, 90: 1.65, 95: 1.96, 99: 2.58, 99.5: 2.807, 99.9: 3.291}
    interval = 95

    plt.figure(1, figsize = [9,7], dpi=800)
    plt.tight_layout()
    plt.subplot(2,1,1)
    x_ax = []
    for j in np.linspace(0, in_sample.shape[0]-1, in_sample.shape[0]):
        x_ax.append(dt.datetime.strptime('2017 8 23', '%Y %m %d') + dt.timedelta(days=j))


    y_true = in_sample['true'].values
    y_pred = in_sample['pred'].values
    y_std = in_sample['std'].values
    plt.plot(x_ax, y_pred, '--', color="red", label="prediction")
    plt.plot(x_ax, y_true, '-', color="black", label="ground_truth")
    plt.fill_between(x_ax,
                     np.squeeze(y_pred - interval_dict[interval] * y_std),
                     np.squeeze(y_pred + interval_dict[interval] * y_std),
                     color="pink", alpha=0.5, label=str(interval)+"% conf")
    plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
    plt.minorticks_on()
    plt.xlim([x_ax[0], x_ax[-1]])
    plt.ylim([0,55])
    plt.ylabel('ili cases', fontsize = 8)
    plt.title('2017/18', fontsize = 8)



    ax1 = plt.subplot(2,1,2)
    ax2 = ax1.twinx()

    x_ax = []
    for j in np.linspace(0, out_of_sample.shape[0]-1, out_of_sample.shape[0]):
        x_ax.append(dt.datetime.strptime('2019 8 23', '%Y %m %d') + dt.timedelta(days=j))

    y_true = out_of_sample['true'].values
    y_pred = out_of_sample['pred'].values
    y_std = out_of_sample['std'].values

    ax1.plot(x_ax, y_pred,'--', color="red", label="prediction")
    ax1.plot(x_ax, y_true, '-',color="black", label="ground_truth")
    ax2.plot(covid['date'].values, covid['covid'].values, '-.',color="green",label='covid')
    ax1.fill_between(x_ax,
                     np.squeeze(y_pred - interval_dict[interval] * y_std),
                     np.squeeze(y_pred + interval_dict[interval] * y_std),
                     color="pink", alpha=0.5, label=str(interval)+"% conf")

    plt.gcf().axes[1].xaxis.set_major_formatter(formatter)
    plt.minorticks_on()

    x_ax = []
    for j in np.linspace(0, in_sample.shape[0] - 1, in_sample.shape[0]):
        x_ax.append(dt.datetime.strptime('2019 8 23', '%Y %m %d') + dt.timedelta(days=j))
    plt.xlim([x_ax[0], x_ax[-1]])

    plt.title('2019/20', fontsize = 8)
    ax1.legend(loc = 'upper left', fontsize = 8)
    ax2.legend(loc = 'upper right', fontsize = 8)
    ax2.set_ylabel('covid deaths', fontsize = 8)
    ax1.set_ylabel('ili cases', fontsize = 8)

    # plt.savefig('/Users/michael/Documents/github/Forecasting/Out_of_Sample_Compare.png')
    plt.show()

