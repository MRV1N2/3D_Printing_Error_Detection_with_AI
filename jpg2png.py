import os
from PIL import Image

folder = "first_test_dataset_yolo\\images"
path = os.path.abspath(os.path.join(os.path.dirname(__file__), folder))

files = os.listdir(path)
for file in files:
    filename = path+"\\"+file
    if '.jpg' in filename:
        Image.open(filename).save(filename.split(".jpg")[0] + ".png")
        os.remove(filename)

print(f"Done. Converted {sum('.jpg' in s for s in files)} files.")