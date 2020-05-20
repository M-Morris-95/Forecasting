import pandas as pd
import matplotlib.pyplot as plt
train = pd.read_csv('training_data.csv')
dropout = pd.read_csv('dropout_uncertainty_predictions.csv')
model = pd.read_csv('model_uncertainty_predictions.csv')

plt.figure(1)
plt.scatter(train.x_train,
            train.y_train,
            marker='+',
            color='green',
            label='Training Data Points')
plt.fill_between(dropout.x_test,
                 dropout.prediction_mean-2*dropout.prediction_std,
                 dropout.prediction_mean+2*dropout.prediction_std,
                 color = 'red',
                 label = 'Dropout confidence interval (2 stddevs)',
                 alpha = 0.3)

plt.plot(dropout.x_test,
         dropout.prediction_mean,
         color = 'red',
         label = 'Dropout Mean Prediction')
plt.fill_between(model.x_test,
                 model.prediction_mean-2*model.prediction_std,
                 model.prediction_mean+2*model.prediction_std,
                 color = 'blue',
                 label = 'VI confidence interval (2 stddevs)',
                 alpha = 0.3)
plt.plot(model.x_test,
         model.prediction_mean,
         color = 'blue',
         label = 'VI Mean Prediction')

plt.plot([-0.5, -0.5], [-2, 2],
         color = 'black',
         linestyle = 'dashed',
         linewidth = 2)
plt.plot([0.5, 0.5], [-2, 2],
         color = 'black',
         linestyle='dashed',
         linewidth = 2)
plt.ylim([-2,2])
plt.xlim([-1.5,1.5])
plt.text(-1, -1.8, 'out of sample', fontsize=10, horizontalalignment='center')
plt.text( 1, -1.8, 'out of sample', fontsize=10, horizontalalignment='center')
plt.text( 0, -1.8, 'training set' , fontsize=10, horizontalalignment='center')
plt.title('Model Confidence Intervals Comparison', fontsize = 10)
plt.xlabel('Input' , fontsize = 10)
plt.ylabel('Output', fontsize = 10)
plt.legend(fontsize=8)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

plt.show()





# plt.figure(2)
# plt.scatter(train.x_train,
#             train.y_train,
#             marker='+',
#             color='blue',
#             label='Training Data Points')
# plt.fill_between(model.x_test,
#                  model.prediction_mean-2*model.prediction_std,
#                  model.prediction_mean+2*model.prediction_std,
#                  color = 'pink',
#                  label = 'confidence interval (2 stddevs)',
#                  alpha = 0.5)
# plt.plot(model.x_test,
#          model.prediction_mean,
#          color = 'red',
#          label = 'mean_prediction')
# plt.title('Model with confidence intervals from variational inference', fontsize = 8)
# plt.xlabel('Input', fontsize = 8)
# plt.ylabel('Output', fontsize = 8)
# plt.legend(fontsize=8)
# plt.grid(b=True, which='major', color='#666666', linestyle='-')
# plt.minorticks_on()
# plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#
# plt.show()