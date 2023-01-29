import os

import joblib
import keras

from config import *
from utils import DataGenerator, read_annotation_lines
from models import Yolov4
import pathlib
import tensorflow as tf

# Hiermit kann man das Ganze auf der CPU laufen lassen
# tf.config.set_visible_devices([], 'GPU')


class Ai_code():

    def __init__(self):

        path = str(pathlib.Path().resolve())
        train_lines = read_annotation_lines(f'{path}\\{DATASET_TRAIN}\\anno.txt')
        val_lines = read_annotation_lines(f'{path}\\{DATASET_VALID}\\anno.txt')
        FOLDER_PATH_TRAIN = f'{path}\\{DATASET_TRAIN}\\images'
        FOLDER_PATH_VALID = f'{path}\\{DATASET_VALID}\\images'
        self.class_name_path = f'{path}\\{DATASET_TRAIN}\\classes.txt'
        self.class_name_path_valid = f'{path}\\{DATASET_VALID}\\classes.txt'
        self.data_gen_train = DataGenerator(train_lines, self.class_name_path, FOLDER_PATH_TRAIN)
        self.data_gen_val = DataGenerator(val_lines, self.class_name_path_valid, FOLDER_PATH_VALID)

        return

    def main(self, hyperparam, callbacks, trial_number=0):

        path = str(pathlib.Path().resolve())

        #  Einstellen der Hyperparameter
        opt_yolo_config = hyperparam['yolo_config']
        #opt_yolo_config['batch_size'] = hyperparam['BATCH_SIZE']
        #opt_yolo_config['iou_loss_thresh'] = hyperparam['IOU_LOSS_THRESH']
        #opt_yolo_config['iou_threshold'] = hyperparam['IOU_THRESHOLD']
        #opt_yolo_config['score_threshold'] = hyperparam['SCORE_THRESHOLD']

        model = Yolov4(weight_path=None,
                       class_name_path=self.class_name_path,
                       config=opt_yolo_config)

        model.fit(self.data_gen_train,
                  initial_epoch=0,
                  epochs=55,
                  val_data_gen=self.data_gen_val,
                  callbacks=callbacks)

        model.yolo_model.save('./yolo_model.h5')
        #model.save_model("./model_default_settings.weigths")
        #model.load_model("./model_default_settings.weigths")
        model.yolo_model = keras.models.load_model('./yolo_model.h5')

        # Model evaluierung
        path = str(pathlib.Path().resolve())

        if not os.path.exists(os.path.join(path, DATASET_TEST, "prediction")): os.mkdir(os.path.join(path, DATASET_TEST, "prediction"))

        for picture in os.listdir(f'{path}\\{DATASET_TEST}\\images'):
            res = model.predict(f'{path}\\{DATASET_TEST}\\images\\{picture}', plot_img=True)
            with open(f"{path}\\{DATASET_TEST}\\prediction\\{picture[:len(picture) - 4]}.txt", 'w') as output_file:

                # Irgendwie war das hier noch nicht richtig fertig...
                # if len(res) > 0:
                #     output_file.write(f'{res["w"][0]} {res["score"][0]} {res["x1"][0]} {res["y1"][0]} {res["x2"][0]} {res["y2"][0]}')

                # Hab es jetzt so umgesetzt:
                if not res.empty:
                    for index, row in res.iterrows():
                        output_file.write(f'{row["class_name"]} {row["score"]} {row["x1"]} {row["y1"]} {row["x2"]} {row["y2"]}\n')

        mAP = model.eval_map(gt_folder_path=f'{path}\\{DATASET_TEST}\\ground-truth',
                             pred_folder_path=f'{path}\\{DATASET_TEST}\\prediction',
                             temp_json_folder_path=f'{path}\\leonHsStuff\\JSON',
                             output_files_path=f'{path}\\leonHsStuff\\OUT')

        return mAP


if __name__ == '__main__':

    # DONT SET THE IMPORT GLOBAL
    from ai_handler import Ai_handler
    Ai_handler()
