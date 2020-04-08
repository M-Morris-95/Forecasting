import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from Parser import GetParser
from Data_Builder import *
from Plotter import *
from Logger import *
from Models import *

parser = GetParser()
args = parser.parse_args()
kl_loss = []
lik_loss = []
EPOCHS, BATCH_SIZE = args.Epochs, args.Batch_Size

logging = logger(args)
models, look_aheads, max_k = logging.get_inputs()

fig = plotter(1)
fig2 = [plotter(2, size=[20,5], dpi=500),  plotter(3, size=[12,10], dpi=500),  plotter(4, size=[12,10], dpi=500),  plotter(5, size=[12,10], dpi=500)]

plot_train=False

for Model in models:
    for look_ahead in look_aheads:
        for k in range(max_k):
            for fold_num in range(1,5):
                print(k, fold_num)
                tf.random.set_seed(0)
                logging.update_details(fold_num=fold_num, k=k, model=Model, look_ahead=look_ahead)
                data = data_builder(args, fold=fold_num, look_ahead=look_ahead)
                x_train, y_train, y_train_index, x_test, y_test, y_test_index = data.build(squared = args.Square_Inputs, normalise_all=True)

                if Model == 'simpleGRU': model = Simple_GRU(x_train, y_train)
                if Model == 'GRU': model = GRU(x_train, y_train)
                if Model == 'ENCODER': model = Encoder(x_train, y_train)
                if Model == 'ATTENTION': model = Attention(x_train, y_train)
                if Model == 'R_ATTN': model = Recurrent_Attention(x_train, y_train)
                if Model == 'LINEAR': model = Linear(x_train, y_train)
                if Model == 'TRANSFORMER': model = Transformer(x_train, y_train)
                if Model == 'GRU_DATA_UNCERTAINTY': model = GRU_Data_Uncertainty(x_train, y_train)
                if Model == 'LINEAR_DATA_UNCERTAINTY': model = Linear_Data_Uncertainty(x_train, y_train)
                if Model == 'LINEAR_MODEL_UNCERTAINTY': model = Linear_Model_Uncertainty(x_train, y_train)
                if Model == 'GRU_MODEL_UNCERTAINTY': model = GRU_Model_Uncertainty(x_train, y_train)
                if Model == 'LINEAR_COMBINED_UNCERTAINTY': model = Linear_Combined_Uncertainty(x_train, y_train, args = args)
                if Model == 'GRU_COMBINED_UNCERTAINTY': model = GRU_Combined_Uncertainty(x_train, y_train, args = args)

                x_train, y_train, x_test, y_test = model.modify_data(x_train, y_train, x_test, y_test)

                model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, plot = True)

                try:
                    kl_loss.append(np.squeeze(np.asarray(model.kl_loss)))
                    lik_loss.append(np.squeeze(np.asarray(model.lik_loss)))
                except:
                    pass

                prediction, stddev, pred, model_mean = model.predict(x_test)

                # plt.scatter(np.tile(np.linspace(0, pred.shape[1]-1, pred.shape[1]), (1, pred.shape[0])), pred.reshape(-1), s=0.02, alpha=0.1, color='blue', label='data uncertainty')
                # plt.scatter(np.tile(np.linspace(0, pred.shape[1]-1, pred.shape[1]), (1, model_mean.shape[0])), model_mean.reshape(-1), s=0.02, alpha=1,
                #             color='red',  label='model uncertainty')
                # plt.plot(y_test, color = 'green', label='true value')
                # plt.plot(np.mean(model_mean, 0), color='yellow', label='predicted value')
                #
                # plt.xlim([0, pred.shape[1]])
                # plt.ylim([-10,50])
                # plt.legend()
                # plt.show()

                if plot_train: fig2[fold_num - 1].plot_conf(fold_num, model.train_prediction, y_train, model.train_stddev,
                                                 split=False)
                fig.plot_conf(fold_num, prediction, y_test, stddev)

                if args.Logging:
                    logging.log(prediction, y_test, model, stddev, save=True, save_weights=False, col_names = data.columns)
if args.Logging:
    logging.save(last=True)
    fig.save(logging.save_directory + '/predictions.png')

try:
    np.save(logging.save_directory + 'likelihood_loss.npy', np.asarray(lik_loss))
    np.save(logging.save_directory + 'KL_divergence_loss.npy', np.asarray(kl_loss))
except:
    pass

fig.show()
# for i in range(4):
#     fig2[i].show()

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(np.asarray(lik_loss)[i,:], label='likelihood')
    plt.plot(np.asarray(kl_loss)[i, :], label='Kl divergence')
    plt.ylim(0,50)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title(str(2013+i)+'/' + str(14+i))
    plt.legend()
plt.show()