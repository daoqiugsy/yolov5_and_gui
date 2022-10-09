import json
from data.convert.pascal_voc_io import PascalVocWriter
import os
from PIL import Image
import shutil
'''
class: 
['traffic light', 'traffic sign', 'car', 'area/drivable', 'lane/road curb', 'person', 'area/alternative',
 'lane/single white', 'lane/double yellow', 'bus', 'truck', 'lane/single yellow', 'lane/crosswalk', 'rider',
  'bike', 'lane/double white', 'motor', 'train', 'lane/single other', 'lane/double other', 'area/unknown']
'''

# cls_name = ['car', 'truck', 'bus']
# cls_name = ['person', 'rider']
cls_name = ['traffic sign']

image_dir = r'Z:\dataset\bbd100k\val\images'
# save_dir = r'Z:\dataset\bbd100k\test\car_annotations'
labels_dir = r'Z:\dataset\bbd100k\val\json_labels'

labels_path = os.listdir(labels_dir)
labels_path = [os.path.join(labels_dir, name) for name in labels_path]

image_save_dir = r'Z:\dataset\bbd100k_traffic_sign\test\images'
annotation_save_dir = r'Z:\dataset\bbd100k_traffic_sign\test\annotations'

for idx, label_path in enumerate(labels_path):

    print(idx)
    with open(label_path, 'r', encoding='utf-8') as fd:
        json_obj = json.load(fd)

    image_name = json_obj['name'] + '.jpg'
    image_path = os.path.join(image_dir, image_name)
    with Image.open(image_path) as im:
        size = im.size

    writer = PascalVocWriter('', image_name, size)
    objs = json_obj['frames'][0]['objects']

    count = 0
    for obj in objs:
        cls = obj['category']
        if cls in cls_name:
            count += 1
            if 'box2d' in obj.keys():
                bbox = [obj['box2d']['x1'], obj['box2d']['y1'], obj['box2d']['x2'], obj['box2d']['y2'], ]
                bbox = [int(c) for c in bbox]
                writer.addBndBox(*bbox, 'person', False)
            else:

                count = -10000
    if count >= 1:
        shutil.copy(image_path, os.path.join(image_save_dir, image_name))
        writer.save(os.path.join(annotation_save_dir, json_obj['name'] + '.xml'))
    # writer.save(os.path.join(save_dir, json_obj['name']+'.xml'))