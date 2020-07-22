from metrics import *
import os
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np
import tensorflow_probability as tfp



x = np.linspace(0, 1, 100)
g_t = np.linspace(0.5, 0.5, 100)
pred = np.linspace(0,1, 100)
std = np.linspace(0.01, 0.5, 100)

likelihood = lambda y, p_y: -p_y.log_prob(y)

fig, ax1 = plt.subplots()
ax1.plot(x, g_t,  label = 'ground truth')
ax1.plot(x,pred, color = 'red', label = 'prediction mean')
ax1.fill_between(x, pred-std, pred+std, color='pink', alpha=0.5, label = 'prediction std')
ax1.set_xlabel('x')
ax1.set_ylabel('y', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.legend()

plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Negative Log Likelihood', color='green')
ax2.set_ylabel('CRPS', color='green')  # we already handled the x-label with ax1
ax2.plot(x, crps_gaussian(g_t, pred, std), color = 'green', alpha = 1)
ax2.plot(x, likelihood(g_t, tfp.distributions.Normal(pred, std)), color = 'blue', alpha = 1)

ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim([-0.6, 0.6])
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

# num = 100
# x = np.linspace(0, 0.5, num) #prediction
# y = np.linspace(0.01, 2, num) #std
# xx, yy = np.meshgrid(x,y)
# gt = np.zeros(yy.shape) #true value
# crps = crps_gaussian(gt, xx, yy) #crps
#
#
# fig = plt.figure(dpi=500)
# ax = fig.gca(projection='3d')
# ax.plot_surface(xx, yy, crps, cmap='jet')
# ax.set_xlabel('MAE')
# ax.set_ylabel('Standard Deviation')
# ax.set_zlabel('CRPS')
#
# ax.view_init(20, 210)
# plt.show()
# # # os.chdir('/Users/michael/Pictures/graphs')
# # # for i in range(0, 360, 20):
# # #     for j in range(0, 360, 20):
# # #         ax.view_init(i, j)
# # #         plt.savefig(str(i) + '_'+str(j)+'.png')
# #
# # likelihood = np.asarray(likelihood(gt, tfp.distributions.Normal(xx, yy))) #NLL
# #
# # fig = plt.figure(dpi=500)
# # ax = fig.gca(projection='3d')
# # ax.plot_surface(xx, yy, likelihood, cmap='jet')
# # ax.set_xlabel('MAE')
# # ax.set_ylabel('Standard Deviation')
# # ax.set_zlabel('NLL')
# #
# # ax.view_init(20, 220)
# # plt.show()
