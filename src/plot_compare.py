import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.dates import DateFormatter
from scipy import special, integrate, stats
import datetime as dt


logging_root = '/Users/michael/Documents/github/Forecasting/' + 'Logging'

# comp1 = '/LINEAR_7LA_May_28_13_28'
# comp2 = '/GRU_MODEL_UNCERTAINTY_7LA_May_28_12_18'
# simons_file = '/Users/michael/Documents/github/data_forecasting_ILI/forecasts/days_ahead_7/ensembles/eng_GRU_MI_MO_ensembles_forecasts'

comp1 = '/LINEAR_14LA_Apr_06_10_09'
comp2 = '/GRU_COMBINED_UNCERTAINTY_14LA_Jun_04_19_14'
simons_file = '/Users/michael/Documents/github/data_forecasting_ILI/forecasts/days_ahead_14/ensembles/eng_GRU_MI_MO_ensembles_forecasts'

# comp1 = '/LINEAR_21LA_May_28_13_30'
# comp2 = '/GRU_MODEL_UNCERTAINTY_21LA_May_28_11_30'
# simons_file = '/Users/michael/Documents/github/data_forecasting_ILI/forecasts/days_ahead_21/ensembles/eng_GRU_MI_MO_ensembles_forecasts'

interval_dict = {50:0.67, 80:1.282,85:1.440, 90:1.65, 95:1.96, 99:2.58, 99.5:2.807,99.9:3.291}

simons_forecasts = pd.read_csv(simons_file, header=None)
pred_name = '/test_predictions.csv'
truth_name = '/test_ground_truth.csv'
idx = []
for i in range(1461):
    if (simons_forecasts[0][i][-5:] == '08-24'):
        idx.append(i)

def rmse(error):
    return np.sqrt(np.mean(np.square(error)))

def mse(error):
    return np.mean(np.square(error))

def mae(error):
    return np.mean(np.abs(error))

def lag(true, pred):
    return np.nanargmax(np.asarray(true))-np.nanargmax(np.asarray(pred))

def quantile_loss(true, pred, std, q):
    max(q*(pred-true), (q-1)*(true-pred))


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



plt.figure(1, figsize = [9,7], dpi=200)
formatter = DateFormatter('%b')

errors_GRU_MI_MO = pd.DataFrame(index=['mae','rmse', 'Corr', 'lag'])
errors_GRU_VI = pd.DataFrame(index=['mae','rmse', 'Corr', 'lag'])
errors_linear = pd.DataFrame(index=['mae','rmse', 'Corr', 'lag'])

interval_dict = {50:0.67, 80:1.282,85:1.440, 90:1.65, 95:1.96, 99:2.58, 99.5:2.807,99.9:3.291}

interval = 90
limits = [0, 25, 30, 25, 55]
for i in range(1,5):
    plt.subplot(2,2,i)

    x_ax = []
    for j in np.linspace(0, 365, 366):
        x_ax.append(dt.datetime.strptime(str(2013+i) + ' 8 23', '%Y %m %d') + dt.timedelta(days=j))

    plt.plot(x_ax, truth.iloc[:, i], '-', color='black', label='ground truth')
    plt.plot(x_ax, linear.iloc[:, i], '-.', color='green', label='linear model')
    plt.plot(x_ax, forecasts[i-1, :], '-.', color = 'blue', label = 'GRU-MI-MO')
    plt.plot(x_ax, mean.iloc[:, i], '--',color = 'red', label = 'GRU-VI-mean')
    plt.fill_between(x_ax,
                     mean.iloc[:, i] + interval_dict[interval]*std.iloc[:, i],
                     mean.iloc[:, i] - interval_dict[interval]*std.iloc[:, i],
                     color = 'pink',
                     label = 'GRU-VI '+str(interval)+'% conf',
                     alpha = 0.5)

    plt.gcf().axes[i-1].xaxis.set_major_formatter(formatter)
    plt.minorticks_on()
    plt.title(str(2013+i) + '/' + str(14+i), fontsize = 8)
    plt.ylabel('ILI rate', fontsize = 8)
    plt.ylim((0, limits[i]))

    if i == 1:
        plt.legend(fontsize = 8)
plt.tight_layout()
# plt.savefig('comparison_14_days_ahead.png')
plt.show()

i = 1
for i in range(1,5):
    plt.subplot(2,2,i)
    se = abs(mean.iloc[:,i].values - truth.iloc[:,i].values)
    sorted = np.argsort(se)
    plt.plot(se[sorted], std.iloc[:,i][sorted])
    plt.xlabel('error')
    plt.ylabel('standard deviation')



plt.show()

_normconst = 1.0 / np.sqrt(2.0 * np.pi)
# Cumulative distribution function of a univariate standard Gaussian
# distribution with zero mean and unit variance.
_normcdf = special.ndtr
def _normpdf(x):
    """Probability density function of a univariate standard Gaussian
    distribution with zero mean and unit variance.
    """
    return _normconst * np.exp(-(x * x) / 2.0)

def crps_gaussian(x, mu, sig, grad=False):
    """
    Computes the CRPS of observations x relative to normally distributed
    forecasts with mean, mu, and standard deviation, sig.
    CRPS(N(mu, sig^2); x)
    Formula taken from Equation (5):
    Calibrated Probablistic Forecasting Using Ensemble Model Output
    Statistics and Minimum CRPS Estimation. Gneiting, Raftery,
    Westveld, Goldman. Monthly Weather Review 2004
    http://journals.ametsoc.org/doi/pdf/10.1175/MWR2904.1
    Parameters
    ----------
    x : scalar or np.ndarray
        The observation or set of observations.
    mu : scalar or np.ndarray
        The mean of the forecast normal distribution
    sig : scalar or np.ndarray
        The standard deviation of the forecast distribution
    grad : boolean
        If True the gradient of the CRPS w.r.t. mu and sig
        is returned along with the CRPS.
    Returns
    -------
    crps : scalar or np.ndarray or tuple of
        The CRPS of each observation x relative to mu and sig.
        The shape of the output array is determined by numpy
        broadcasting rules.
    crps_grad : np.ndarray (optional)
        If grad=True the gradient of the crps is returned as
        a numpy array [grad_wrt_mu, grad_wrt_sig].  The
        same broadcasting rules apply.
    """
    x = np.asarray(x)
    mu = np.asarray(mu)
    sig = np.asarray(sig)
    # standadized x
    sx = (x - mu) / sig
    # some precomputations to speed up the gradient
    pdf = _normpdf(sx)
    cdf = _normcdf(sx)
    pi_inv = 1. / np.sqrt(np.pi)
    # the actual crps
    crps = sig * (sx * (2 * cdf - 1) + 2 * pdf - pi_inv)
    if grad:
        dmu = 1 - 2 * cdf
        dsig = 2 * pdf - pi_inv
        return crps, np.array([dmu, dsig])
    else:
        return crps

for i in range(1,5):
    print(np.nanmean(crps_gaussian(truth.iloc[:,i],mean.iloc[:,i],std.iloc[:,i])))

for i in range(1, 5):
    simons_error = truth.iloc[:,i] - forecasts[i - 1, :]
    my_error = truth.iloc[:,i] - mean.iloc[:, i]


    errors_GRU_MI_MO[str(2013 + i) + '/' + str(14 + i)] = [mae(simons_error),
                                                        rmse(simons_error),
                                                        np.corrcoef(truth.iloc[:-1,i], forecasts[i - 1, :-1])[0,1],
                                                        lag(truth.iloc[:-1,i], forecasts[i - 1, :-1])]
    errors_GRU_VI[str(2013 + i) + '/' + str(14 + i)] = [mae(my_error),
                                                        rmse(my_error),
                                                        np.corrcoef(truth.iloc[:-1,i] , mean.iloc[:-1, i])[0,1],
                                                        lag(truth.iloc[:-1,i] , mean.iloc[:-1, i])]
    linear_error = truth.iloc[:, i] - linear.iloc[:, i]
    errors_linear[str(2013 + i) + '/' + str(14 + i)] = [mae(linear_error),
                                                        rmse(linear_error),
                                                        np.corrcoef(truth.iloc[:-1,i] , linear.iloc[:-1, i])[0,1],
                                                        lag(truth.iloc[:-1,i] , linear.iloc[:, i])]






# errors_GRU_MI_MO['avg'] = np.mean(np.abs(errors_GRU_MI_MO.values), 1)
#
# errors_GRU_VI['avg'] = np.mean(np.abs(errors_GRU_VI.values), 1)
#
# errors_linear['avg'] = np.mean(np.abs(errors_linear.values), 1)
# print('GRU-MI-MO')
# print(errors_GRU_MI_MO.round(3).to_latex())
# print('GRU-VI')
# print(errors_GRU_VI.round(3).to_latex())
# print('LINEAR')
# print(errors_linear.round(3).to_latex())

