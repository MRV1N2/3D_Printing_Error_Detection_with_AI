import sys
FOLDER_PATH = 'YOUR PATH'
sys.path.append(FOLDER_PATH)

import cv2
import numpy as np
import os
from models import Yolov4

model = Yolov4(class_name_path = os.path.join(FOLDER_PATH, 'class_names', 'coco_classes.txt'), weight_path='yolov4.weights')

model.predict(os.path.join(FOLDER_PATH, 'img', 'test.jpg'), random_color=True)