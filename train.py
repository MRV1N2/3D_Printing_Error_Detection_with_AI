from utils import DataGenerator, read_annotation_lines
from models import Yolov4
import pathlib

class Ai_code():

    def __init__(self):

        path = str(pathlib.Path().resolve())
        train_lines, val_lines = read_annotation_lines(f'{path}\\first_test_dataset\\anno.txt', test_size=0.1)
        FOLDER_PATH = f'{path}\\first_test_dataset\\images'
        class_name_path = f'{path}\\first_test_dataset\\classes.txt'
        self.data_gen_train = DataGenerator(train_lines, class_name_path, FOLDER_PATH)
        self.data_gen_val = DataGenerator(val_lines, class_name_path, FOLDER_PATH)

        return

    def main(self, hyperparam, callbacks, trial_number=0):

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

        #Todo Model Evaluation

        # model.predict ...

        Score = -1
        return Score

if __name__ == '__main__':

    # DONT SET THE IMPORT GLOBAL
    from ai_handler import Ai_handler
    Ai_handler()