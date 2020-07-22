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

fig = plotter(1)
loss_fig = plotter(3)
fig2 = [plotter(2, size=[20,5], dpi=500),  plotter(3, size=[12,10], dpi=500),  plotter(4, size=[12,10], dpi=500),  plotter(5, size=[12,10], dpi=500)]

plot_train=False
out_of_sample=False

for fold_num in range(4,5):
    print(fold_num)
    tf.random.set_seed(0)
    logging.update_details(fold_num=fold_num, k=0, model=args.Model, look_ahead=args.Look_Ahead)
    data = data_builder(args, fold=fold_num, out_of_sample=False)
    x_train, y_train, x_test, y_test = data.build(normalise_all=False)

    if args.Model == 'simpleGRU': model = Simple_GRU(x_train, y_train, dropout=0, mimo=True)
    if args.Model == 'GRU': model = GRU(x_train, y_train)
    if args.Model == 'ENCODER': model = Encoder(x_train, y_train)
    if args.Model == 'ATTENTION': model = Attention(x_train, y_train)
    if args.Model == 'R_ATTN': model = Recurrent_Attention(x_train, y_train)
    if args.Model == 'LINEAR': model = Linear(x_train, y_train)
    if args.Model == 'TRANSFORMER': model = Transformer(x_train, y_train)
    if args.Model == 'GRU_DATA_UNCERTAINTY': model = GRU_Data_Uncertainty(x_train, y_train)
    if args.Model == 'LINEAR_DATA_UNCERTAINTY': model = Linear_Data_Uncertainty(x_train, y_train)
    if args.Model == 'LINEAR_MODEL_UNCERTAINTY': model = Linear_Model_Uncertainty(x_train, y_train, args=args)
    if args.Model == 'GRU_MODEL_UNCERTAINTY': model = GRU_Model_Uncertainty(x_train, y_train, args=args)
    if args.Model == 'LINEAR_COMBINED_UNCERTAINTY': model = Linear_Combined_Uncertainty(x_train, y_train, args = args)
    if args.Model == 'GRU_COMBINED_UNCERTAINTY': model = GRU_Combined_Uncertainty(x_train, y_train, args = args)
    x_train, y_train, x_test, y_test = model.modify_data(x_train, y_train, x_test, y_test)

    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, plot = False)
    print('trained')
    prediction, stddev, pred, model_mean = model.predict(x_test)
    print('done predictions')

    if args.MIMO:
        prediction = prediction[:, args.MIMO]
        y_test = y_test[:, args.MIMO]
        stddev = stddev[:, args.MIMO]

    logging.log(prediction, y_test, model, stddev, save=True, save_weights=False)

    if plot_train:
        fig2[fold_num - 1].plot_conf(fold_num, model.train_prediction, y_train, model.train_stddev,
                                     split=False)

    fig.plot_conf(fold_num, prediction, y_test, stddev)

    if out_of_sample:
        data2 = data_builder(args, fold=fold_num, look_ahead=look_ahead, out_of_sample=True)
        x_train, y_train, y_train_index, x_test, y_test, y_test_index = data2.build(squared = args.Square_Inputs, normalise_all=True)
        prediction, stddev, pred, model_mean = model.predict(x_test)

        # log_pred(prediction, y_test, fold_num, stddev=stddev)


if args.Logging:
    logging.save(last=True)
    fig.save(logging.save_directory + '/predictions.png')
    loss_fig.save(logging.save_directory + '/loss.png')

print(logging.train_stats)
fig.show()
loss_fig.show()

if out_of_sample:
    data = data_builder(args, fold=fold_num, look_ahead=look_ahead, out_of_sample=False)
    x_train, y_train, y_train_index, x_test, y_test, y_test_index = data.build(squared=args.Square_Inputs,
                                                                               normalise_all=True)
    x_train, y_train, x_test, y_test = model.modify_data(x_train, y_train, x_test, y_test)
    prediction, stddev, pred, model_mean = model.predict(x_test)
    in_sample = pd.DataFrame({'true':y_test, 'pred':prediction, 'std':stddev})
    in_sample.to_csv('/Users/michael/Documents/datasets/in_sample'+str(fold_num)+'.csv')
    fig.plot_conf(1, prediction, y_test, stddev)

    data = data_builder(args, fold=fold_num, look_ahead=look_ahead, out_of_sample=True)
    x_train, y_train, y_train_index, x_test, y_test, y_test_index = data.build(squared=args.Square_Inputs,
                                                                               normalise_all=True)
    x_train, y_train, x_test, y_test = model.modify_data(x_train, y_train, x_test, y_test)
    prediction, stddev, pred, model_mean = model.predict(x_test)
    out_of_sample = pd.DataFrame({'true':y_test, 'pred':prediction, 'std':stddev})
    out_of_sample.to_csv('/Users/michael/Documents/datasets/out_of_sample'+str(fold_num)+'.csv')
    fig.plot_conf(2, prediction, y_test, stddev)
    fig.show()

    print('normalised uncertainty is :      out of sample {:2.2%}      in sample {:2.2%}'.format(out_of_sample['pred'].mean()/out_of_sample['true'].max(),in_sample['pred'].mean()/in_sample['true'].max()))



#
# data = data_builder(args, fold=fold_num, look_ahead=look_ahead, out_of_sample=True)
# x_train, y_train, _, x_test_oos, y_test_oos, _ = data.build(squared=args.Square_Inputs,
#                                                                            normalise_all=True)
# x_train, y_train, x_test_oos, y_test_oos = model.modify_data(x_train, y_train, x_test_oos, y_test_oos)
#
# data = data_builder(args, fold=fold_num, look_ahead=look_ahead, out_of_sample=False)
# x_train, y_train, _, x_test_is, y_test_is, _ = data.build(squared=args.Square_Inputs,
#                                                                            normalise_all=True)
#
# x_train, y_train, x_test_is, y_test_is = model.modify_data(x_train, y_train, x_test_is, y_test_is)
#
# oos_pred_mean, oos_pred_std, _, _ = model.predict(x_test_oos)
# likelihood_oos = tfp.distributions.Normal(loc=oos_pred_mean, scale=oos_pred_std).prob(y_test_oos).numpy()
#
# is_pred_mean, is_pred_std, _, _ = model.predict(x_test_is)
# likelihood_is = tfp.distributions.Normal(loc=is_pred_mean, scale=is_pred_std).prob(y_test_is).numpy()
#
# tr_pred_mean, tr_pred_std, _, _ = model.predict(x_train)
# likelihood_tr = tfp.distributions.Normal(loc=tr_pred_mean, scale=tr_pred_std).prob(y_train).numpy()
#
# likelihood_is.mean()
# likelihood_oos.mean()
# likelihood_tr.mean()
#
#
# print('likelihood - training: {:2.1%}, in-sample test: {:2.1%}, out-of-sample test: {:2.1%}'.format(likelihood_tr.mean(), likelihood_is.mean(), likelihood_oos.mean()))
#
# print(np.mean(np.abs((oos_pred_mean-y_test_oos)/oos_pred_std)))
# print(np.mean(np.abs((is_pred_mean-y_test_is)/is_pred_std)))
# print(np.mean(np.abs((tr_pred_mean-y_train)/tr_pred_std)))