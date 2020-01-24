import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

root = '/Users/michael/Documents/ili_data/Weather/'


folders = ['mean', 'min', 'max']
temp_data = pd.DataFrame(pd.date_range(pd.datetime(1878, 1, 1), periods=51864, freq='1D'))
for fold in folders:
    os.chdir(root + fold)
    list = os.listdir()
    list.sort()

    temp = pd.read_csv(list[0], sep="\s+",
                       names=['year', 'day', 'jan', 'feb', 'march', 'april', 'may', 'june', 'july', 'aug', 'sep', 'oct',
                              'nov', 'dec'])

    if fold == 'mean':
        temp = temp[3286:]


    temp = temp.drop('year', axis=1)
    temp = temp.drop('day', axis=1)

    temp = np.asarray(temp)
    num_years = int(len(temp)/31)
    temp = np.split(temp, num_years)
    temp = np.asarray(temp)
    temp = temp.reshape((num_years, -1), order='F')
    temp = temp.reshape(-1)
    temp = temp[temp != -999]


    temp_data[str(fold)] = temp/10
    print('next')

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

plt.plot(running_mean(temp_data['mean'].values, 100))
plt.show()





