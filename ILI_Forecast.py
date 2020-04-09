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
EPOCHS, BATCH_SIZE = args.Epochs, args.Batch_Size

logging = logger(args)
models, look_aheads, max_k = logging.get_inputs()

fig = plotter(1)
loss_fig = plotter(3)
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

                prediction, stddev, pred, model_mean = model.predict(x_test)

                logging.log(prediction, y_test, model, stddev, save=True, save_weights=False, col_names=data.columns)

                if plot_train:
                    fig2[fold_num - 1].plot_conf(fold_num, model.train_prediction, y_train, model.train_stddev,
                                                 split=False)

                fig.plot_conf(fold_num, prediction, y_test, stddev)
                loss_fig.plot_loss(fold_num, logging.model_history)

if args.Logging:
    logging.save(last=True)
    fig.save(logging.save_directory + '/predictions.png')
    loss_fig.save(logging.save_directory + '/loss.png')

fig.show()
loss_fig.show()