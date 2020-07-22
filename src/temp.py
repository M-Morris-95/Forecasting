import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import datetime as dt

data = pd.read_csv('/Users/michael/Downloads/coronavirus-cases_latest.csv')

locations = ['Richmond upon Thames']
for location in locations:

    idx = np.argwhere(data['Area name'] == location)
    try:
        idx = idx.reshape(-1, 2)
    except:
        pass

    idx = idx[:, 0]

    n_days = abs(dt.datetime.strptime(data.iloc[idx]['Specimen date'].iloc[0], '%Y-%m-%d') - dt.datetime.strptime(
        data.iloc[idx]['Specimen date'].iloc[-1], '%Y-%m-%d')).days

    x_ax = []
    for j in range(n_days+1):
        x_ax.append((dt.datetime.strptime(data.iloc[idx]['Specimen date'].iloc[-1], '%Y-%m-%d') + dt.timedelta(days=j)).date())

    new_data = pd.DataFrame(index=np.asarray(x_ax), columns=['cases'], data = np.zeros(n_days+1))

    for j in range(idx.shape[0]):
        pos = np.argwhere(dt.datetime.strptime(data.iloc[idx].iloc[j]['Specimen date'], '%Y-%m-%d').date() == new_data.index)[0][0]
        new_data.iloc[pos]['cases'] = data.iloc[idx[j]]['Daily lab-confirmed cases']

    plt.plot(new_data.index, new_data.cases,
                 label = location)


formatter = DateFormatter('%b-%d')

plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
# plt.yticks(np.arange(0, 25, step=1))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel('date')
plt.ylabel('daily new cases')
plt.legend()

plt.show()