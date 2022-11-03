import os

import joblib
import pandas as pd

from utils import DataGenerator, read_annotation_lines
from models import Yolov4
import pathlib


class Ai_code():

    def __init__(self):

        path = str(pathlib.Path().resolve())
        train_lines, val_lines = read_annotation_lines(f'{path}\\first_test_dataset\\anno.txt', test_size=0.1)
        FOLDER_PATH = f'{path}\\first_test_dataset\\images'
        self.class_name_path = f'{path}\\first_test_dataset\\classes.txt'
        self.data_gen_train = DataGenerator(train_lines, self.class_name_path, FOLDER_PATH)
        self.data_gen_val = DataGenerator(val_lines, self.class_name_path, FOLDER_PATH)

        return

    def main(self, hyperparam, callbacks, trial_number=0):

        #  Einstellen der Hyperparameter
        opt_yolo_config = hyperparam['yolo_config']
        opt_yolo_config['batch_size'] = hyperparam['BATCH_SIZE']
        opt_yolo_config['iou_loss_thresh'] = hyperparam['IOU_LOSS_THRESH']
        opt_yolo_config['iou_threshold'] = hyperparam['IOU_THRESHOLD']
        opt_yolo_config['score_threshold'] = hyperparam['SCORE_THRESHOLD']

        model = Yolov4(weight_path=None,
                       class_name_path=self.class_name_path,
                       config=opt_yolo_config)

        #Todo Pruner Implemention
        model.fit(self.data_gen_train,
                  initial_epoch=0,
                  epochs=10,
                  val_data_gen=self.data_gen_val,
                  callbacks=callbacks)

        model.save_model("./model.weigths")
        model.load_model("./model.weigths")

        # Todo Model Evaluation
        # Sorry Marvin, ich habe keine Ahnung, was ich hier tue :'(
        path = str(pathlib.Path().resolve())
        for picture in os.listdir(f'{path}\\first_test_dataset_yolo\\images'):
            res = model.predict(f'{path}\\first_test_dataset_yolo\\images\\{picture}')
            with open(f"{path}\\first_test_dataset_yolo\\prediction\\{picture.split('.', 1)[0]}.txt", 'w') as output_file:
                if len(res) > 0:
                    output_file.write(f'{res["w"][0]} {res["score"][0]} {res["x1"][0]} {res["y1"][0]} {res["x2"][0]} {res["y2"][0]}')

        mAP = model.eval_map(gt_folder_path=f'{path}\\first_test_dataset_yolo\\labels',
                       pred_folder_path=f'{path}\\first_test_dataset_yolo\\prediction',
                       temp_json_folder_path=f'{path}\\leonHsStuff\\JSON',
                       output_files_path=f'{path}\\leonHsStuff\\OUT')

        # for file in (test_directory):
        #   merken was auf dem Bild zu sehen ist -> w
        #   e = model.predict ...
        #   Vergleichen, was hat die KI gesagt (e) und was war wirklich zu sehen (w)
        #   tabellen in python: numpy.array 2D oder pandas.Dataframe

        return mAP


if __name__ == '__main__':

    # DONT SET THE IMPORT GLOBAL
    from ai_handler import Ai_handler
    Ai_handler()
