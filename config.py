
# Optuna Settings
STUDY_NAME = 'Examination_Hyperparameters_10'
OPTUNA_DATABASE_PATH = 'sqlite:///ai_3d_printing_error_detection.db'

# Hyperparameters
#TODO Wich Hyperparameters could we improve in this project
BATCH_SIZE = 1
IOU_LOSS_THRESH = [0.1, 0.9]
IOU_THRESHOLD = [0.1, 0.9]
SCORE_THRESHOLD = [0.1,0.9]

# This Yolo config,will be overwritten by optuna
# ToDo find script to set anchors
yolo_config = {
    # Basic
    'img_size': (416, 416, 3),
    'anchors': [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
    'strides': [8, 16, 32],
    'xyscale': [1.2, 1.1, 1.05],

    # Training
    'iou_loss_thresh': 0.5,
    'batch_size': 1, # 8,
    'num_gpu': 1,  # 2,

    # Inference
    'max_boxes': 100,
    'iou_threshold': 0.413,
    'score_threshold': 0.3,
}

# DONT REMOVE THIS:
CONFIG_VARS = vars()
