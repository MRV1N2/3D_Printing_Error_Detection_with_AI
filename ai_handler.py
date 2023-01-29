from tensorflow import keras
from types import ModuleType
from pathlib import Path
from train import Ai_code
import tensorflow as tf
from config import *
import optuna
import shutil
import sys
import os

class Ai_handler(Ai_code):

    def __init__(self):

        self._ai_code = Ai_code()

        path = str(os.path.abspath(__file__))
        path = path[:path.rfind('/')]
        path = path[:path.rfind('/')]
        if not os.path.exists(f'{path}/doc'):
            os.makedirs(f'{path}/doc')
        self._doc_path = f'{path}/doc'

        # Read the config.py and sort the variables
        self._config = {}
        self._hparams = {}
        for key, value in CONFIG_VARS.items():
            if not '__' in key and not isinstance(value, ModuleType) and not 'CONFIG_VARS' in key:
                if isinstance(value, list):
                    self._hparams[key] = value
                else:
                    self._config[key] = value

        self._last_best_study_value = 100000
        self._last_best_study_id = 0
        self._study = optuna.create_study(direction="maximize",
                                # directions=["maximize", "maximize"],
                                # pruner=optuna.pruners.SuccessiveHalvingPruner(),
                                study_name=STUDY_NAME,
                                storage=OPTUNA_DATABASE_PATH,
                                load_if_exists=True)

        self._study.optimize(self._objective, n_trials=99999999999, n_jobs=1, show_progress_bar=False)


    def _objective(self, trial):
        # Optuna part to tell which variables from the config.py can be optimized and how.
        # Here should never have to change anything.

        trial_hparams = {}
        opt_hparams = {}
        trial_hparams.update(self._config)
        trial_hparams.update(opt_hparams)

        for key, value in self._hparams.items():

            # Int
            if type(value[0]) == int and len(value) == 2:
                opt_hparams[key] = trial.suggest_int(key, value[0], value[1])
                trial.set_user_attr(key, opt_hparams[key])

            # Float
            elif type(value[0]) == float:
                opt_hparams[key] = trial.suggest_float(key, value[0], value[1])
                trial.set_user_attr(key, opt_hparams[key])

            # Categorical
            else:
                try:
                    # Check if every value has a legal datatype
                    for singlevalue in value:
                        if not isinstance(singlevalue, str) and not isinstance(singlevalue, int) and not isinstance(singlevalue, float):
                            raise Exception
                    opt_hparams[key] = trial.suggest_categorical(key, value)
                    trial.set_user_attr(key, opt_hparams[key])
                except:

                    # Other datatyps or structures
                    name_list = []
                    for i in range(len(value)):
                        if isinstance(value[i], str) or isinstance(value[i], int) or isinstance(value[i], float):
                            if 'MODEL' in key:
                                name_list.append(opt_hparams['MODEL'].__name__)
                            else:
                                name_list.append(f'{key} {value[i]}')
                        else:
                            name_list.append(f'{key} {i}. (nameless)')
                    try_name = trial.suggest_categorical(key, name_list)
                    opt_hparams[key] = value[name_list.index(try_name)]


                    if not 'MODEL' in key:
                        trial.set_user_attr(key, try_name)
                    else:
                        stringlist = []
                        trial_hparams.update(self._config)
                        trial_hparams.update(opt_hparams)
                        input_shape = (trial_hparams['MEASUREMENT_WINDOW_LEN'],
                                                trial_hparams['MEASUREMENT_SCANSAMPLES'],
                                                trial_hparams['THIRD_INPUT_DIMENSION_SIZE'])
                        try:
                            opt_hparams[key](input_shape).summary(print_fn=lambda x: stringlist.append(x))
                        except:
                            temp_model = opt_hparams[key](input_shape)
                            temp_model.build((None, *input_shape))
                            temp_model.summary(print_fn=lambda x: stringlist.append(x))

                        for i in range(len(stringlist)):
                            line = stringlist[i]
                            if not '__' in line and not '==' in line:
                                linenr = str((i + 1000))
                                trial.set_system_attr(linenr, line)

        try:
            if not 'MODEL' in list(opt_hparams.keys()):
                opt_hparams['MODEL'] = MODEL
        except:
            pass

        #  Create the documentation about this trial

        if True:
            #try:
            if True:
                # Set the labels of the trial and their contents in the tensorboard dashboard
                self._logdir = f'./doc/02_tensorboard_logs/{STUDY_NAME}_trial_nr_{trial.number}'
                self._file_writer = tf.summary.create_file_writer(self._logdir)
                with self._file_writer.as_default():

                    i = 0
                    for key, value in opt_hparams.items():
                        if not 'MODEL' in key:
                            tf.summary.text("Configuration", f'{key}: {str(value)}', step=i)
                            i+=1
                        else:
                            stringlist = []
                            trial_hparams.update(self._config)
                            trial_hparams.update(opt_hparams)
                            input_shape = (trial_hparams['MEASUREMENT_WINDOW_LEN'],
                                                trial_hparams['MEASUREMENT_SCANSAMPLES'],
                                                trial_hparams['THIRD_INPUT_DIMENSION_SIZE'])
                            try:
                                opt_hparams[key](input_shape).summary(print_fn=lambda x: stringlist.append(x))
                            except:
                                temp_model = opt_hparams[key](input_shape)
                                temp_model.build((None, *input_shape))
                                temp_model.summary(print_fn=lambda x: stringlist.append(x))

                            stringlist = list(reversed(stringlist))
                            for o in range(len(stringlist)):
                                if not '__' in stringlist[o] and not '==' in stringlist[o]:
                                    tf.summary.text("model.summary", stringlist[o], step=i)
                                    i+=1
                    self._file_writer.flush()

                # Train the model
                with tf.summary.create_file_writer(self._logdir).as_default():

                    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./models/', save_weights_only=True, verbose=1, monitor='loss')
                    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=self._logdir, histogram_freq=1,profile_batch=0)
                    op_callback = Optuna_Callback(trial)
                    early_stopping = tf.keras.callbacks.EarlyStopping()
                    callbacks = [tb_callback]

                    trial_hparams.update(self._config)
                    trial_hparams.update(opt_hparams)
                    rmse = self._ai_code.main(trial_hparams, callbacks, trial.number)

                    tf.summary.scalar('val_rmse', rmse, step=1)

                # Only keep the best attempt in this study in Tensorboard
                if rmse > self._last_best_study_value and False:
                    shutil.rmtree(f'./doc/02_tensorboard_logs/{STUDY_NAME}_trial_nr_{trial.number}') #TODO: Soll das _ogs heißen?
                else:
                    self._last_best_study_value = rmse
                    self._last_best_study_id = trial.number

                return rmse

            #except Exception as e:
            #    print(f"Exception at objective: {e}, {sys.exc_info()}")
            #    return 10


class Optuna_Callback(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        y = logs["loss"]
        x = epoch
        self.trial.report(y, step=x)

    def __init__(self, trial):
        self.trial = trial
