import video_detection
import os
import cv2
import pathlib
import config


video_folder_path = os.getcwd()

# print(video_folder_path)

detector = video_detection.VideoDetector()

detector.loadModel(classname_path=os.path.join(str(pathlib.Path().resolve()), "..", config.DATASET_TRAIN, "classes.txt"),
                   model_path="../yolo_model.h5")

# image = "Capture-d-ecran--580-_jpg.rf.536a3776d40f4ec00fd13fd42e2727cf.jpg"

# res = detector.get_Model().predict(os.path.join(str(pathlib.Path().resolve()), "..", config.DATASET_TEST, "images", image), random_color=True, plot_img=True)

# raw_img = cv2.imread(os.path.join(str(pathlib.Path().resolve()), "..", config.DATASET_TEST, "images", image))[:, :, ::-1]

# while True:
#     # detected_frame = detector.get_Model().predict_img_raw(raw_img)
#     detected_frame, output_objects_array = detector.detectObjectsFromImage(
#         input_image=raw_img, input_type="array", output_type="array",
#         minimum_percentage_probability=10,
#         display_percentage_probability=True,
#         display_object_name=True, test=True)
#
#     cv2.imshow('Frame', detected_frame)
#     if cv2.waitKey(1) == 27:
#         break  # esc to quit

# detector.detectObjectsFromVideo(input_file_path=os.path.join(video_folder_path, "test_bilder.mp4"),
#                                 output_file_path=os.path.join(video_folder_path, "output"),
#                                 frames_per_second=20,
#                                 minimum_percentage_probability=4,
#                                 log_progress=True,
#                                 display_video=True,
#                                 test=True)

camera = cv2.VideoCapture(1)

detector.detectObjectsFromVideo(camera_input=camera,
                                output_file_path=os.path.join(video_folder_path, "video_output"),
                                save_detected_video=False,
                                frame_detection_interval=1,
                                minimum_percentage_probability=10,
                                log_progress=True,
                                display_video=True,
                                test=True)
