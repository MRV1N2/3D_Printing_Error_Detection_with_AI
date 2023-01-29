import os
from models import Yolov4
import keras
import config
import pathlib

model = Yolov4(weight_path="./yolo_model.h5",
               class_name_path=os.path.join(str(pathlib.Path().resolve()), config.DATASET_TRAIN, "classes.txt"),
               config=config.yolo_config)

# model.load_model()

image = "772a27e26920431770e781dda062398b-image-asset_img_5eb0af2d092e0_jpg.rf.31b414878e18130aadb448a6fefae357.jpg"

res = model.predict(os.path.join(str(pathlib.Path().resolve()), config.DATASET_TRAIN, "images", image), random_color=True, plot_img=True)

# res = model.predict(os.path.join(str(pathlib.Path().resolve()), "video", "1.jpeg"), random_color=True, plot_img=True)

# print(res)
