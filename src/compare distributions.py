import numpy as np
from metrics import *
import pandas as pd
import tensorflow_probability as tfp


likelihood = lambda y, p_y: -p_y.log_prob(y)

logging_root = '/Users/michael/Documents/github/Forecasting/Logging'

roots = ['/GRU_COMBINED_UNCERTAINTY_14LA_Jun_04_19_14',
         '/GRU_MODEL_UNCERTAINTY_14LA_Jul_02_15_25',
         '/GRU_DATA_UNCERTAINTY_14LA_Jun_17_13_22',
         '/DROPOUT_UNCERTAINTY_14LA_Jun_0.2',
         '/DROPOUT_UNCERTAINTY_14LA_Jul_0.5',
         '/LINEAR_COMBINED_UNCERTAINTY_14LA_Apr_06_09_53']

names = ['GRU combined', 'GRU model', 'GRU data', 'GRU dropout 0.2','GRU dropout 0.5', 'lin combined']


CRPS = pd.DataFrame()
MAE = pd.DataFrame()
Likelihood = pd.DataFrame()

for j, root in enumerate(roots):
    mean = pd.read_csv(logging_root + root + '/test_predictions.csv')
    std = pd.read_csv(logging_root + root + '/test_stddev.csv')
    # std.iloc[:,1] = std.iloc[:,1]*1e-5
    truth = pd.read_csv(logging_root + root + '/test_ground_truth.csv')
    t_crps = []
    t_mae = []
    t_lik = []
    for i in range(1,5):
        t_crps.append(np.nanmean(crps_gaussian(truth.iloc[:,i],mean.iloc[:,i],std.iloc[:,i])))
        t_mae.append(np.mean(np.abs(truth.iloc[:,i]-mean.iloc[:,i])))
        likelihood(truth.iloc[:,i], tfp.distributions.Normal(mean.iloc[:, i], std.iloc[:, i]))
        t_lik.append(np.nanmean(likelihood(truth.iloc[:,i], tfp.distributions.Normal(mean.iloc[:, i], std.iloc[:, i]))))
        t_crps[-1] = 100*t_crps[-1]/np.max(truth.iloc[:,i])
        t_mae[-1] = 100*t_mae[-1] / np.max(truth.iloc[:, i])
        t_lik[-1] = 100*t_lik[-1] / np.max(truth.iloc[:, i])
    t_crps.append(np.mean(t_crps))
    t_mae.append(np.mean(t_mae))
    t_lik.append(np.mean(t_lik))
    CRPS[names[j]] = np.asarray(t_crps)
    MAE[names[j]] = np.asarray(t_mae)
    Likelihood[names[j]]=np.asarray(t_lik)

CRPS.index = ['2014/15','2015/16','2016/17','2017/18','avg']
MAE.index = ['2014/15','2015/16','2016/17','2017/18','avg']
Likelihood.index = ['2014/15','2015/16','2016/17','2017/18','avg']
CRPS = CRPS.round(2)
MAE = MAE.round(2)
Likelihood = Likelihood.round(2)

pd.set_option('display.max_columns', 6)
pd.set_option('display.width', 200)
print('Lower is better')
print('                                 CRPS                                ')
print('---------------------------------------------------------------------')
print(CRPS)
print('---------------------------------------------------------------------')
print('')
print('')
print('                                 MAE                                 ')
print('---------------------------------------------------------------------')
print(MAE)
print('---------------------------------------------------------------------')
print('')
print('')
print('                      NEGATIVE LOG LIKELIHOOD                        ')
print('---------------------------------------------------------------------')
print(Likelihood)
print('---------------------------------------------------------------------')