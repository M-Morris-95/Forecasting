import pandas as pd
import numpy as np
import metrics
import os
import time
import tensorflow as tf

class logger:
    def __init__(self, args):
        timestamp = time.strftime('_%b_%d_%H_%M', time.localtime())
        self.root_directory = os.getcwd()
        self.ret_max_k = args.K
        self.do_stddev = False

        self.save_model = args.Save_Model

        if args.K != 1:
            self.iter = True
        else:
            self.iter = False
        if args.Model == 'ALL':
            self.indvidual_models = True
            self.ret_model = ['GRU', 'ATTENTION', 'ENCODER']
        else:
            self.indvidual_models = False
            self.model = args.Model
            self.ret_model = [args.Model]

        if args.Look_Ahead == 0:
            self.indvidual_look_ahead = True
            self.ret_look_ahead = [7, 14, 21]
            look_ahead_str = ''
        else:
            self.indvidual_look_ahead = False
            self.look_ahead = args.Look_Ahead
            self.ret_look_ahead = np.asarray([args.Look_Ahead])
            look_ahead_str = '_' + str(args.Look_Ahead) + 'LA'

        if not args.Server:
            self.logging_directory = '/Users/michael/Documents/github/Forecasting/Logging/'
        else:
            self.logging_directory = '/home/mimorris/Forecasting/Logging/'
        self.save_directory = self.logging_directory + args.Model + look_ahead_str + timestamp

        self.train_stats = pd.DataFrame(index=['MAE', 'RMSE', 'R', 'Lag'])
        self.test_predictions = pd.DataFrame()
        self.test_ground_truth = pd.DataFrame()
        self.stddev = pd.DataFrame()
        self.cleanup()
        self.first=True

    def get_inputs(self):
        return self.ret_model, self.ret_look_ahead, self.ret_max_k

    def update_details(self, fold_num, model=None, look_ahead=None, k=None, epochs=None):
        self.fold_num = fold_num
        fold_str = str(2013 + fold_num) + '/' + str(14 + fold_num)

        if self.indvidual_models:
            model_str = str(model)
        else:
            model_str = ''

        if self.indvidual_look_ahead:
            look_ahead_str = '_' + str(look_ahead) + '_'
        else:
            look_ahead_str = ''

        if self.iter:
            iter_str = '_' + str(k)
        else:
            iter_str = ''

        if epochs:
            epochs_str = '_' + str(epochs)
        else:
            epochs_str = ''

        self.save_name = model_str + look_ahead_str + fold_str + iter_str + epochs_str

    def log_pred(self, y_pred, y_true, fold, stddev=None):
        if self.first:
            self.first = False
            self.out_of_sample = pd.DataFrame()

        self.out_of_sample['true' + str(fold)] = y_pred
        self.out_of_sample['pred' + str(fold)] = y_true
        self.out_of_sample['std'+str(fold)] = stddev
        self.out_of_sample.to_csv('out_of_sample_pred.csv')

    def log(self, y_pred, y_true, model, stddev=None,save=False, save_weights=False, col_names=None):

        self.model_history = pd.DataFrame(model.model.history.history)
        if y_pred.ndim == 3:
            y_pred = np.squeeze(y_pred)
        if y_true.ndim == 3:
            y_true = np.squeeze(y_true)
        if y_true.ndim == 2:
            y_true = y_true[:, -1]
        if y_pred.ndim == 2:
            y_pred = y_pred[:, -1]

        if stddev is not None:
            self.do_stddev = True
            if stddev.ndim > 1:
                if stddev.shape[1] != 0:
                    stddev = stddev[:, -1]
                else:
                    stddev = np.squeeze(stddev)

            stddev = np.append(stddev, np.nan)[:366]
            self.stddev[str(self.save_name)] = stddev

        y_pred = np.asarray(y_pred)
        self.train_stats[str(self.save_name)] = metrics.evaluate(y_true, y_pred)

        y_true = np.append(y_true, np.nan)[:366]
        y_pred = np.append(y_pred, np.nan)[:366]

        self.test_ground_truth[str(self.save_name)] = y_true
        self.test_predictions[str(self.save_name)] = y_pred

        try:
            self.model_history = pd.DataFrame.from_dict(model.log)
        except:
            self.model_history = pd.DataFrame(model.model.history.history)


        # if save_weights:
        #     final_weights = model.weights[0].numpy()[-167:]
        #     weights = pd.DataFrame(columns=['weight'],index = np.asarray(col_names), data = np.squeeze(final_weights))
        #     weights.to_csv('weights.csv')

        if save:
            self.save(model)

    def cleanup(self):
        root = self.logging_directory
        folders = list(os.walk(root))[1:]

        for folder in folders:
            folder[1]
            # folder example: ('FOLDER/3', [], ['file'])
            if (len(folder[2]) == 0) and (len(folder[1]) == 0):
                os.rmdir(folder[0])

    def save(self, model=None, last = False):
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
        os.chdir(self.save_directory)

        if not os.path.exists(self.save_directory + '/models'):
            os.makedirs(self.save_directory + '/models')
        os.chdir(self.save_directory + '/models')

        self.model_history.to_csv(r'' + self.save_name.replace('/', '_') + '.csv')

        if model is not None:
            if self.save_model:
                # model.model.save(self.save_name.replace('/', '_'), save_format='tf')
                try:
                    model.model.save(self.save_name.replace('/', '_'), save_format='h5')
                    model.model.save_weight
                    # tf.keras.models.save_model(model.model, self.save_name.replace('/', '_') + '.hdf5')
                except:
                    pass
            os.chdir(self.save_directory)



        if self.do_stddev:
            self.stddev.to_csv(r'test_stddev.csv')

        if last:
            self.train_stats['Average'] = np.asarray([np.mean(self.train_stats.iloc[0].values),
                np.mean(self.train_stats.iloc[1].values),
                np.mean(self.train_stats.iloc[2].values),
                np.mean(np.abs(self.train_stats.iloc[3].values))])
            self.train_stats = self.train_stats.transpose()

            self.train_stats.iloc[:, 0] = self.train_stats.iloc[:, 0].round(3)
            self.train_stats.iloc[:, 1] = self.train_stats.iloc[:, 1].round(3)
            self.train_stats.iloc[:, 2] = self.train_stats.iloc[:, 2].round(3)
            self.train_stats.iloc[:, 3] = self.train_stats.iloc[:, 3].round(0)

        os.chdir(self.save_directory)
        self.train_stats.to_csv(r'train_stats.csv')
        self.test_predictions.to_csv(r'test_predictions.csv')
        self.test_ground_truth.to_csv(r'test_ground_truth.csv')
        os.chdir(self.root_directory)