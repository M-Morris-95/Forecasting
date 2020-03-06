import numpy as np
import matplotlib.pyplot as plt
TRUE_STDDEV = 0.3
TRUE_MEAN = 0.3

PRED_STDDEV = 5
PRED_MEAN = 5

max_iter = 30

dataset = np.zeros((1,max_iter))
dataset = np.full(max_iter, np.nan)
mean = []
std = []

for i in range(max_iter):
    dataset[i] = np.random.normal(TRUE_MEAN, TRUE_STDDEV)
    PRED_STDDEV = np.nanstd(dataset)
    PRED_MEAN = np.nanmean(dataset)

    mean.append(PRED_MEAN)
    std.append(PRED_STDDEV)

    # print('mean = ',PRED_MEAN, 'stddev = ', PRED_STDDEV)

plt.plot(mean)
plt.plot(std)
plt.show()

p_sun_blowing_up = 0.0000001
p_rolls = 1/36
p_rolls_if_sun_blew_up = 5/36

prob_table = np.zeros((2**6, 6))
for i in range(6):
    for j in range(2**6):
        if np.mod(j+1,2**(i+1)) <= 2**i:
            prob_table[j, i] = 1

axis = []
a = []
lim = 6
max = 10000
for i in range(max):
    x = (i/(max/lim))-(lim/2)
    a.append(np.log(1+np.exp(x)))
    axis.append(x)
plt.plot(axis, a)
plt.show()
