# https://github.com/Cartucho/mAP/blob/master/scripts/extra/convert_gt_xml.py

import sys
import os
import glob
import xml.etree.ElementTree as ET

folder = "online_dataset\\trainAndValid"

# make sure that the cwd() in the beginning is the location of the python script (so that every path makes sense)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# change directory to the one with the files to be changed
# parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# parent_path = os.path.abspath(os.path.join(parent_path, os.pardir))
GT_PATH = os.path.join(folder, "ground-truth")
ANNO_PATH = os.path.join(folder, 'Annotations')
print(ANNO_PATH)

if not os.path.exists(GT_PATH):
  os.makedirs(GT_PATH)


# create VOC format files
xml_list = glob.glob(ANNO_PATH + '\\*.xml')
if len(xml_list) == 0:
    print("Error: no .xml files found in ground-truth")
    sys.exit()

for tmp_file in xml_list:
    #print(tmp_file)
    # 1. create new file (VOC format)
    new_f_name = tmp_file.replace(ANNO_PATH, GT_PATH).replace(".xml", ".txt")
    with open(new_f_name, "a") as new_f:
        print(new_f_name)
        root = ET.parse(tmp_file).getroot()
        for obj in root.findall('object'):
            obj_name = obj.find('name').text
            bndbox = obj.find('bndbox')
            left = bndbox.find('xmin').text
            top = bndbox.find('ymin').text
            right = bndbox.find('xmax').text
            bottom = bndbox.find('ymax').text
            new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
    # 2. move old file (xml format) to backup
    # os.rename(tmp_file, os.path.join("backup", tmp_file))
print("Conversion completed!")