# Professionell geklaut von https://github.com/OlafenwaMoses/ImageAI/blob/fe2d6bab3ddb1027c54abe7eb961364928869a30/imageai/Detection/Custom/__init__.py
import cv2
import os
import re
import numpy as np
import config
from bbox import BoundBox
from models import Yolov4
import pathlib
import keras


class VideoDetector:

    def __init__(self):
        # self.__model_type = ""
        # self.__model_path = ""
        self.__model_labels = ["fail"]
        # self.__model_anchors = []
        # self.__detection_config_json_path = ""
        self.__input_size = 416
        self.__object_threshold = 0.4
        self.__nms_threshold = 0.4
        self.__model = None
        self.__detection_utils = CustomDetectionUtils(labels=self.__model_labels)

    def loadModel(self, classname_path, model_path):
        self.__model = Yolov4(weight_path=model_path,
                              class_name_path=classname_path,
                              config=config.yolo_config)
        # self.__model.yolo_model = keras.models.load_model(model_path)

    def get_Model(self):
        return self.__model

    def detectObjectsFromImage(self, input_image="", output_image_path="", input_type="file", output_type="file",
                               extract_detected_objects=False, minimum_percentage_probability=50, nms_treshold=0.4,
                               display_percentage_probability=True, display_object_name=True, thread_safe=False, test=False):
        """
        'detectObjectsFromImage()' function is used to detect objects observable in the given image:
                    * input_image , which can be a filepath or image numpy array in BGR
                    * output_image_path (only if output_type = file) , file path to the output image that will contain the detection boxes and label, if output_type="file"
                    * input_type (optional) , filepath/numpy array of the image. Acceptable values are "file" and "array"
                    * output_type (optional) , file path/numpy array/image file stream of the image. Acceptable values are "file" and "array"
                    * extract_detected_objects (optional) , option to save each object detected individually as an image and return an array of the objects' image path.
                    * minimum_percentage_probability (optional, 30 by default) , option to set the minimum percentage probability for nominating a detected object for output.
                    * nms_threshold (optional, o.45 by default) , option to set the Non-maximum suppression for the detection
                    * display_percentage_probability (optional, True by default), option to show or hide the percentage probability of each object in the saved/returned detected image
                    * display_display_object_name (optional, True by default), option to show or hide the name of each object in the saved/returned detected image
                    * thread_safe (optional, False by default), enforce the loaded detection model works across all threads if set to true, made possible by forcing all Keras inference to run on the default graph
            The values returned by this function depends on the parameters parsed. The possible values returnable
            are stated as below
            - If extract_detected_objects = False or at its default value and output_type = 'file' or
                at its default value, you must parse in the 'output_image_path' as a string to the path you want
                the detected image to be saved. Then the function will return:
                1. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name (string)
                    * percentage_probability (float)
                    * box_points (list of x1,y1,x2 and y2 coordinates)
            - If extract_detected_objects = False or at its default value and output_type = 'array' ,
              Then the function will return:
                1. a numpy array of the detected image
                2. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name (string)
                    * percentage_probability (float)
                    * box_points (list of x1,y1,x2 and y2 coordinates)
            - If extract_detected_objects = True and output_type = 'file' or
                at its default value, you must parse in the 'output_image_path' as a string to the path you want
                the detected image to be saved. Then the function will return:
                1. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name (string)
                    * percentage_probability (float)
                    * box_points (list of x1,y1,x2 and y2 coordinates)
                2. an array of string paths to the image of each object extracted from the image
            - If extract_detected_objects = True and output_type = 'array', the the function will return:
                1. a numpy array of the detected image
                2. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name (string)
                    * percentage_probability (float)
                    * box_points (list of x1,y1,x2 and y2 coordinates)
                3. an array of numpy arrays of each object detected in the image
        :param input_image:
        :param output_image_path:
        :param input_type:
        :param output_type:
        :param extract_detected_objects:
        :param minimum_percentage_probability:
        :param nms_treshold:
        :param display_percentage_probability:
        :param display_object_name:
        :param thread_safe:
        :return image_frame:
        :return output_objects_array:
        :return detected_objects_image_array:
        """

        if False:  # self.__model is None:
            raise ValueError("You must call the loadModel() function before making object detection.")
        else:

            if output_type == "file":
                # from the image file, lets keep the directory and the filename, but remove its  format
                # if output_image_path is path/to/the/output/image.png
                # then output_image_folder is  path/to/the/output/image
                # let's check if it is in the appropriated format soon to fail early
                output_image_folder, n_subs = re.subn(r'\.(?:jpe?g|png|tif|webp|PPM|PGM)$', '', output_image_path,
                                                      flags=re.I)
                if n_subs == 0:
                    # if no substitution was done, the given output_image_path is not in a supported format,
                    # raise an error
                    raise ValueError("output_image_path must be the path where to write the image. "
                                     "Therefore it must end as one the following: "
                                     "'.jpg', '.png', '.tif', '.webp', '.PPM', '.PGM'. {} found".format(
                        output_image_path))
                elif extract_detected_objects:
                    # Results must be written as files and need to extract detected objects as images,
                    # let's create a folder to store the object's images
                    objects_dir = output_image_folder + "-objects"

                    os.makedirs(objects_dir, exist_ok=True)

            self.__object_threshold = minimum_percentage_probability / 100
            self.__nms_threshold = nms_treshold

            output_objects_array = []
            detected_objects_image_array = []

            if input_type == "file":
                image = cv2.imread(input_image)
            elif input_type == "array":
                image = input_image
            else:
                raise ValueError("input_type must be 'file' or 'array'. {} found".format(input_type))

            if not test:  # self.__model_type == "yolov3":
                image_frame = image.copy()

                height, width, channels = image.shape

                image = cv2.resize(image, (self.__input_size, self.__input_size))

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image = image.astype("float32") / 255.

                # expand the image to batch
                image = np.expand_dims(image, 0)



                boxes = list()

                if thread_safe == True:
                    # with K.get_session().graph.as_default():
                    yolo_results = self.__model.predict_img_raw(image)
                else:
                    yolo_results = self.__model.predict_img_raw(image)

                for idx, result in enumerate(yolo_results):
                    box_set = self.__detection_utils.decode_netout(result[0], config.yolo_config['anchors'],
                                                                   # self.__model_anchors[idx],
                                                                   self.__object_threshold, self.__input_size,
                                                                   self.__input_size)
                    boxes += box_set

                self.__detection_utils.correct_yolo_boxes(boxes, height, width, self.__input_size,
                                                          self.__input_size)

                self.__detection_utils.do_nms(boxes, self.__nms_threshold)

                all_boxes, all_labels, all_scores = self.__detection_utils.get_boxes(boxes, self.__model_labels,
                                                                                     self.__object_threshold)

                for object_box, object_label, object_score in zip(all_boxes, all_labels, all_scores):
                    each_object_details = dict()
                    each_object_details["name"] = object_label
                    each_object_details["percentage_probability"] = object_score

                    if object_box.xmin < 0:
                        object_box.xmin = 0
                    if object_box.ymin < 0:
                        object_box.ymin = 0

                    each_object_details["box_points"] = [object_box.xmin, object_box.ymin, object_box.xmax,
                                                         object_box.ymax]
                    output_objects_array.append(each_object_details)

                drawn_image = self.__detection_utils.draw_boxes_and_caption(image_frame.copy(), all_boxes, all_labels,
                                                                            all_scores, show_names=display_object_name,
                                                                            show_percentage=display_percentage_probability)

                if extract_detected_objects:

                    for cnt, each_object in enumerate(output_objects_array):

                        splitted_image = image_frame[each_object["box_points"][1]:each_object["box_points"][3],
                                         each_object["box_points"][0]:each_object["box_points"][2]]
                        if output_type == "file":
                            splitted_image_path = os.path.join(objects_dir, "{}-{:05d}.jpg".format(each_object["name"],
                                                                                                   cnt))

                            cv2.imwrite(splitted_image_path, splitted_image)
                            detected_objects_image_array.append(splitted_image_path)
                        elif output_type == "array":
                            detected_objects_image_array.append(splitted_image.copy())

                if output_type == "file":
                    # we already validated that the output_image_path is a supported by OpenCV one
                    cv2.imwrite(output_image_path, drawn_image)

                if extract_detected_objects:
                    if output_type == "file":
                        return output_objects_array, detected_objects_image_array
                    elif output_type == "array":
                        return drawn_image, output_objects_array, detected_objects_image_array

                else:
                    if output_type == "file":
                        return output_objects_array
                    elif output_type == "array":
                        return drawn_image, output_objects_array
            else:
                drawn_image = self.__model.predict_img_raw(image)
                return drawn_image, output_objects_array

    def detectObjectsFromVideo(self, input_file_path="", camera_input=None, output_file_path="", frames_per_second=20,
                               frame_detection_interval=1, minimum_percentage_probability=50, log_progress=False,
                               display_percentage_probability=True, display_object_name=True, save_detected_video=True,
                               per_frame_function=None, per_second_function=None, per_minute_function=None,
                               video_complete_function=None, return_detected_frame=False, detection_timeout=None,
                               display_video=False, test=False):
        """
        'detectObjectsFromVideo()' function is used to detect objects observable in the given video path or a camera input:
            * input_file_path , which is the file path to the input video. It is required only if 'camera_input' is not set
            * camera_input , allows you to parse in camera input for live video detections
            * output_file_path , which is the path to the output video. It is required only if 'save_detected_video' is not set to False
            * frames_per_second , which is the number of frames to be used in the output video
            * frame_detection_interval (optional, 1 by default)  , which is the intervals of frames that will be detected.
            * minimum_percentage_probability (optional, 50 by default) , option to set the minimum percentage probability for nominating a detected object for output.
            * log_progress (optional) , which states if the progress of the frame processed is to be logged to console
            * display_percentage_probability (optional), can be used to hide or show probability scores on the detected video frames
            * display_object_name (optional), can be used to show or hide object names on the detected video frames
            * save_save_detected_video (optional, True by default), can be set to or not to save the detected video
            * per_frame_function (optional), this parameter allows you to parse in a function you will want to execute after each frame of the video is detected. If this parameter is set to a function, after every video  frame is detected, the function will be executed with the following values parsed into it:
                -- position number of the frame
                -- an array of dictinaries, with each dictinary corresponding to each object detected. Each dictionary contains 'name', 'percentage_probability' and 'box_points'
                -- a dictionary with with keys being the name of each unique objects and value are the number of instances of the object present
                -- If return_detected_frame is set to True, the numpy array of the detected frame will be parsed as the fourth value into the function
            * per_second_function (optional), this parameter allows you to parse in a function you will want to execute after each second of the video is detected. If this parameter is set to a function, after every second of a video is detected, the function will be executed with the following values parsed into it:
                -- position number of the second
                -- an array of dictionaries whose keys are position number of each frame present in the last second , and the value for each key is the array for each frame that contains the dictionaries for each object detected in the frame
                -- an array of dictionaries, with each dictionary corresponding to each frame in the past second, and the keys of each dictionary are the name of the number of unique objects detected in each frame, and the key values are the number of instances of the objects found in the frame
                -- a dictionary with its keys being the name of each unique object detected throughout the past second, and the key values are the average number of instances of the object found in all the frames contained in the past second
                -- If return_detected_frame is set to True, the numpy array of the detected frame will be parsed
                                                                    as the fifth value into the function
            * per_minute_function (optional), this parameter allows you to parse in a function you will want to execute after each minute of the video is detected. If this parameter is set to a function, after every minute of a video is detected, the function will be executed with the following values parsed into it:
                -- position number of the minute
                -- an array of dictionaries whose keys are position number of each frame present in the last minute , and the value for each key is the array for each frame that contains the dictionaries for each object detected in the frame
                -- an array of dictionaries, with each dictionary corresponding to each frame in the past minute, and the keys of each dictionary are the name of the number of unique objects detected in each frame, and the key values are the number of instances of the objects found in the frame
                -- a dictionary with its keys being the name of each unique object detected throughout the past minute, and the key values are the average number of instances of the object found in all the frames contained in the past minute
                -- If return_detected_frame is set to True, the numpy array of the detected frame will be parsed as the fifth value into the function
            * video_complete_function (optional), this parameter allows you to parse in a function you will want to execute after all of the video frames have been detected. If this parameter is set to a function, after all of frames of a video is detected, the function will be executed with the following values parsed into it:
                -- an array of dictionaries whose keys are position number of each frame present in the entire video , and the value for each key is the array for each frame that contains the dictionaries for each object detected in the frame
                -- an array of dictionaries, with each dictionary corresponding to each frame in the entire video, and the keys of each dictionary are the name of the number of unique objects detected in each frame, and the key values are the number of instances of the objects found in the frame
                -- a dictionary with its keys being the name of each unique object detected throughout the entire video, and the key values are the average number of instances of the object found in all the frames contained in the entire video
            * return_detected_frame (optionally, False by default), option to obtain the return the last detected video frame into the per_per_frame_function, per_per_second_function or per_per_minute_function
            * detection_timeout (optionally, None by default), option to state the number of seconds of a video that should be detected after which the detection function stop processing the video
        :param input_file_path:
        :param camera_input:
        :param output_file_path:
        :param frames_per_second:
        :param frame_detection_interval:
        :param minimum_percentage_probability:
        :param log_progress:
        :param display_percentage_probability:
        :param display_object_name:
        :param save_detected_video:
        :param per_frame_function:
        :param per_second_function:
        :param per_minute_function:
        :param video_complete_function:
        :param return_detected_frame:
        :param detection_timeout:
        :param display_video:
        :param test:
        :return output_video_filepath:
        :return counting:
        :return output_objects_array:
        :return output_objects_count:
        :return detected_copy:
        :return this_second_output_object_array:
        :return this_second_counting_array:
        :return this_second_counting:
        :return this_minute_output_object_array:
        :return this_minute_counting_array:
        :return this_minute_counting:
        :return this_video_output_object_array:
        :return this_video_counting_array:
        :return this_video_counting:
        """

        output_frames_dict = {}
        output_frames_count_dict = {}


        if (camera_input != None):
            input_video = camera_input
        else:
            input_video = cv2.VideoCapture(input_file_path)

        output_video_filepath = output_file_path + '.avi'

        frame_width = int(input_video.get(3))
        frame_height = int(input_video.get(4))
        output_video = cv2.VideoWriter(output_video_filepath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                       frames_per_second,
                                       (frame_width, frame_height))

        counting = 0
        predicted_numbers = None
        scores = None
        detections = None

        detection_timeout_count = 0
        video_frames_count = 0

        if True:  # (self.__model_type == "yolov3"):

            while (input_video.isOpened()):
                ret, frame = input_video.read()

                if (ret == True):

                    # detected_frame = frame.copy()

                    video_frames_count += 1
                    if (detection_timeout != None):
                        if ((video_frames_count % frames_per_second) == 0):
                            detection_timeout_count += 1

                        if (detection_timeout_count >= detection_timeout):
                            break

                    output_objects_array = []

                    counting += 1

                    if (log_progress == True):
                        print("Processing Frame : ", str(counting))

                    check_frame_interval = counting % frame_detection_interval

                    if (counting == 1 or check_frame_interval == 0):

                        detected_frame, output_objects_array = self.detectObjectsFromImage(
                                input_image=frame, input_type="array", output_type="array",
                                minimum_percentage_probability=minimum_percentage_probability,
                                display_percentage_probability=display_percentage_probability,
                                display_object_name=display_object_name, test=test)

                    if not test:
                        output_frames_dict[counting] = output_objects_array

                        output_objects_count = {}
                        for eachItem in output_objects_array:
                            eachItemName = eachItem["name"]
                            try:
                                output_objects_count[eachItemName] = output_objects_count[eachItemName] + 1
                            except:
                                output_objects_count[eachItemName] = 1

                        output_frames_count_dict[counting] = output_objects_count

                    if (save_detected_video == True):
                        output_video.write(detected_frame)

                    if display_video:
                        detected_frame = cv2.resize(detected_frame, (832, 832))
                        cv2.imshow('Frame', detected_frame)
                        if cv2.waitKey(1) == 27:
                            break  # esc to quit

                    if (counting == 1 or check_frame_interval == 0):
                        if (per_frame_function != None):
                            if (return_detected_frame == True):
                                per_frame_function(counting, output_objects_array, output_objects_count,
                                                   detected_frame)
                            elif (return_detected_frame == False):
                                per_frame_function(counting, output_objects_array, output_objects_count)

                    if (per_second_function != None):
                        if (counting != 1 and (counting % frames_per_second) == 0):

                            this_second_output_object_array = []
                            this_second_counting_array = []
                            this_second_counting = {}

                            for aa in range(counting):
                                if (aa >= (counting - frames_per_second)):
                                    this_second_output_object_array.append(output_frames_dict[aa + 1])
                                    this_second_counting_array.append(output_frames_count_dict[aa + 1])

                            for eachCountingDict in this_second_counting_array:
                                for eachItem in eachCountingDict:
                                    try:
                                        this_second_counting[eachItem] = this_second_counting[eachItem] + \
                                                                         eachCountingDict[eachItem]
                                    except:
                                        this_second_counting[eachItem] = eachCountingDict[eachItem]

                            for eachCountingItem in this_second_counting:
                                this_second_counting[eachCountingItem] = int(
                                    this_second_counting[eachCountingItem] / frames_per_second)

                            if (return_detected_frame == True):
                                per_second_function(int(counting / frames_per_second),
                                                    this_second_output_object_array, this_second_counting_array,
                                                    this_second_counting, detected_frame)

                            elif (return_detected_frame == False):
                                per_second_function(int(counting / frames_per_second),
                                                    this_second_output_object_array, this_second_counting_array,
                                                    this_second_counting)

                    if (per_minute_function != None):

                        if (counting != 1 and (counting % (frames_per_second * 60)) == 0):

                            this_minute_output_object_array = []
                            this_minute_counting_array = []
                            this_minute_counting = {}

                            for aa in range(counting):
                                if (aa >= (counting - (frames_per_second * 60))):
                                    this_minute_output_object_array.append(output_frames_dict[aa + 1])
                                    this_minute_counting_array.append(output_frames_count_dict[aa + 1])

                            for eachCountingDict in this_minute_counting_array:
                                for eachItem in eachCountingDict:
                                    try:
                                        this_minute_counting[eachItem] = this_minute_counting[eachItem] + \
                                                                         eachCountingDict[eachItem]
                                    except:
                                        this_minute_counting[eachItem] = eachCountingDict[eachItem]

                            for eachCountingItem in this_minute_counting:
                                this_minute_counting[eachCountingItem] = int(
                                    this_minute_counting[eachCountingItem] / (frames_per_second * 60))

                            if (return_detected_frame == True):
                                per_minute_function(int(counting / (frames_per_second * 60)),
                                                    this_minute_output_object_array, this_minute_counting_array,
                                                    this_minute_counting, detected_frame)

                            elif (return_detected_frame == False):
                                per_minute_function(int(counting / (frames_per_second * 60)),
                                                    this_minute_output_object_array, this_minute_counting_array,
                                                    this_minute_counting)


                else:
                    break

            if (video_complete_function != None):

                this_video_output_object_array = []
                this_video_counting_array = []
                this_video_counting = {}

                for aa in range(counting):
                    this_video_output_object_array.append(output_frames_dict[aa + 1])
                    this_video_counting_array.append(output_frames_count_dict[aa + 1])

                for eachCountingDict in this_video_counting_array:
                    for eachItem in eachCountingDict:
                        try:
                            this_video_counting[eachItem] = this_video_counting[eachItem] + \
                                                            eachCountingDict[eachItem]
                        except:
                            this_video_counting[eachItem] = eachCountingDict[eachItem]

                for eachCountingItem in this_video_counting:
                    this_video_counting[eachCountingItem] = this_video_counting[
                                                                eachCountingItem] / counting

                video_complete_function(this_video_output_object_array, this_video_counting_array,
                                        this_video_counting)

            input_video.release()
            output_video.release()

            if (save_detected_video == True):
                return output_video_filepath


class CustomDetectionUtils:
    def __init__(self, labels):
        self.__labels = labels
        self.__colors = []

        for i in range(len(labels)):
            color_space_values = np.random.randint(50, 255, size=(3,))
            red, green, blue = color_space_values
            red, green, blue = int(red), int(green), int(blue)
            self.__colors.append([red, green, blue])

    @staticmethod
    def _sigmoid(x):
        return 1. / (1. + np.exp(-x))

    def decode_netout(self, netout, anchors, obj_thresh, net_h, net_w):
        grid_h, grid_w = netout.shape[:2]
        nb_box = 3
        netout = netout.reshape((grid_h, grid_w, nb_box, -1))
        nb_class = netout.shape[-1] - 5
        boxes = []
        netout[..., :2] = self._sigmoid(netout[..., :2])
        netout[..., 4:] = self._sigmoid(netout[..., 4:])
        netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
        netout[..., 5:] *= netout[..., 5:] > obj_thresh

        for row in range(grid_h):
            for col in range(grid_w):
                for b in range(nb_box):
                    # 4th element is objectness score
                    objectness = netout[row, col, b, 4]

                    if objectness <= obj_thresh:
                        continue

                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row, col, b, :4]
                    x = (col + x) / grid_w  # center position, unit: image width
                    y = (row + y) / grid_h  # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height
                    # last elements are class probabilities
                    classes = netout[row, col, b, 5:]
                    box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)
                    boxes.append(box)

        return boxes

    @staticmethod
    def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
        new_w, new_h = net_w, net_h
        for i in range(len(boxes)):
            x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
            y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h
            boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
            boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
            boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
            boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

    def _interval_overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2, x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2, x4) - x3

    def bbox_iou(self, box1, box2):
        intersect_w = self._interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = self._interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
        intersect = intersect_w * intersect_h
        w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
        w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
        union = w1 * h1 + w2 * h2 - intersect

        try:
            result = float(intersect) / float(union)
            return result
        except:
            return 0.0

    def do_nms(self, boxes, nms_thresh):
        if len(boxes) > 0:
            nb_class = len(boxes[0].classes)
        else:
            return

        for c in range(nb_class):
            sorted_indices = np.argsort([-box.classes[c] for box in boxes])

            for i in range(len(sorted_indices)):
                index_i = sorted_indices[i]

                if boxes[index_i].classes[c] == 0: continue

                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if self.bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                        boxes[index_j].classes[c] = 0

    def get_boxes(self, boxes, labels, thresh):
        v_boxes, v_labels, v_scores = list(), list(), list()
        # enumerate all boxes
        for box in boxes:
            # enumerate all possible labels
            for i in range(len(labels)):
                # check if the threshold for this label is high enough
                if box.classes[i] > thresh:
                    v_boxes.append(box)
                    v_labels.append(labels[i])
                    v_scores.append(box.classes[i] * 100)
                # don't break, many labels may trigger for one box
        return v_boxes, v_labels, v_scores

    def label_color(self, label):
        """ Return a color from a set of predefined colors. Contains 80 colors in total.
        Args
            label: The label to get the color for.
        Returns
            A list of three values representing a RGB color.
            If no color is defined for a certain label, the color green is returned and a warning is printed.
        """
        if label < len(self.__colors):
            return self.__colors[label]
        else:
            return 0, 255, 0

    def draw_boxes_and_caption(self, image_frame, v_boxes, v_labels, v_scores, show_names=False, show_percentage=False):

        for i in range(len(v_boxes)):
            box = v_boxes[i]
            y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
            width, height = x2 - x1, y2 - y1
            class_color = self.label_color(self.__labels.index(v_labels[i]))

            image_frame = cv2.rectangle(image_frame, (x1, y1), (x2, y2), class_color, 2)

            label = ""
            if show_names and show_percentage:
                label = "%s : %.3f" % (v_labels[i], v_scores[i])
            elif show_names:
                label = "%s" % (v_labels[i])
            elif show_percentage:
                label = "%.3f" % (v_scores[i])

            if show_names or show_percentage:
                b = np.array([x1, y1, x2, y2]).astype(int)
                cv2.putText(image_frame, label, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (200, 0, 0), 3)
                cv2.putText(image_frame, label, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

        return image_frame
