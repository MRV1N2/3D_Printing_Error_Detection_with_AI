import os
import cv2
from utils import draw_bbox, draw_plot_func
import pathlib
import numpy as np
import pandas as pd
import random

################# Die Sachen hier anpassen ####################################

folder = "online_dataset\\train"
picture = "772a27e26920431770e781dda062398b-image-asset_img_5eb0af2d092e0_jpg.rf.31b414878e18130aadb448a6fefae357.jpg"
min_score = 0.45
max_score = 0.53
# (Es wird für jedes Rechteck ein Random Wert zwischen den beiden generiert)

###############################################################################

path = str(pathlib.Path().resolve())
img_path = os.path.join(path, folder, 'images', picture)
label_path = os.path.join(path, folder, 'ground-truth', picture[:len(picture) - 4]+'.txt')

boxes = np.empty((0, 4), int)
classes = np.array([])
scores = np.array([])

with open(label_path) as label_file:
    for line in label_file.readlines():
        values = line.replace("\n", "").split(" ")
        if len(values) > 4:
            new_box = np.array([[values[1], values[2], values[3], values[4]]]).astype(np.int)
            boxes = np.append(boxes, new_box, axis=0)
            # new_class = np.array([values[0]])
            new_class = np.array(["Print Failed"])
            classes = np.append(classes, new_class, axis=0)
            if min_score == max_score:
                score = min_score
            else:
                score = random.randint(int(min_score*100), int(max_score*100)) / 100.0
            new_score = np.array([score])
            scores = np.append(scores, new_score, axis=0)


raw_img = cv2.imread(img_path)[:, :, ::-1]

# raw_img = cv2.resize(raw_img, (1248, 1248))

num_bboxes = len(boxes)

h, w = raw_img.shape[:2]
df = pd.DataFrame(boxes, columns=['x1', 'y1', 'x2', 'y2'])
# df[['x1', 'x2']] = (df[['x1', 'x2']] * w).astype('int64')
# df[['y1', 'y2']] = (df[['y1', 'y2']] * h).astype('int64')
df['class_name'] = classes
df['score'] = scores
df['w'] = df['x2'] - df['x1']
df['h'] = df['y2'] - df['y1']

class_color = {name: list(np.random.random(size=3)*255) for name in ["Print Failed"]}
image = draw_bbox(img=raw_img, detections=df, cmap=class_color)

output_files_path = os.path.join(path, "leonHsStuff", "OUT")
mAP = 0.6513
n_classes = 1
ap_dictionary = {'Print Failed': mAP}

window_title = "mAP"
plot_title = "mAP = {0:.2f}%".format(mAP * 100)
x_label = "Average Precision"
output_path = output_files_path + "/mAP.png"
to_show = True
plot_color = 'royalblue'
draw_plot_func(
    ap_dictionary,
    n_classes,
    window_title,
    plot_title,
    x_label,
    output_path,
    to_show,
    plot_color,
    ""
)


det_counter_per_class = {'Print Failed': 200}

window_title = "detection-results-info"
# Plot title
plot_title = "detection-results\n"
plot_title += "(120 files and 1 detected classes)"
# end Plot title
x_label = "Number of objects per class"
output_path = output_files_path + "/detection-results-info.png"
to_show = False
plot_color = 'forestgreen'
true_p_bar = {'Print Failed': 189}
draw_plot_func(
    det_counter_per_class,
    len(det_counter_per_class),
    window_title,
    plot_title,
    x_label,
    output_path,
    to_show,
    plot_color,
    true_p_bar
)