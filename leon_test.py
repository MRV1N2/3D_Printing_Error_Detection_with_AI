import pathlib

import pandas as pd

dummy = {
    "x1": [0.2],
    "y1": [0.2],
    "x2": [0.2],
    "y2": [0.2],
    "class_name": [0],
    "score": [0.99],
    "w": [1],
    "h": [1]
}
res = pd.DataFrame(dummy)
del dummy
path = str(pathlib.Path().resolve())
with open(f"{path}\\first_test_dataset_yolo\\prediction\\testFile.txt", 'w') as output_file:
    output_file.write(f'{res["w"][0]} {res["score"][0]} {res["x1"][0]} {res["y1"][0]} {res["x2"][0]} {res["y2"][0]}')