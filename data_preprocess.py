import os
import xml.etree.ElementTree as ET
from config import classes_num, DATA_PATH


def convert_annotation(image_id, f):
    in_file = os.path.join(
        DATA_PATH, f'VOCdevkit/VOC2007/Annotations/{image_id}.xml')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        classes = list(classes_num.keys())
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
             int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        f.write(' ' + ','.join([str(a) for a in b]) + ',' + str(cls_id))


def data_preprocess():
    for image_set in ['train', 'val', 'trainval']:
        print(image_set)
        with open(os.path.join(DATA_PATH, 'VOCdevkit/VOC2007/ImageSets/Main/%s.txt' % (image_set)), 'r') as f:
            image_ids = f.read().strip().split()
        with open(os.path.join(DATA_PATH, "VOCdevkit", '%s.txt' % (image_set)), 'w') as f:
            for image_id in image_ids:
                f.write(os.path.join(
                    DATA_PATH, '%s/VOC2007/JPEGImages/%s.jpg' % ("VOCdevkit", image_id)))
                convert_annotation(image_id, f)
                f.write('\n')


if __name__ == '__main__':
    data_preprocess()
