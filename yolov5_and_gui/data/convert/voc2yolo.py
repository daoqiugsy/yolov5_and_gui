from data.convert.yolo_io import YOLOWriter, YoloReader
from data.convert.pascal_voc_io import PascalVocReader, PascalVocWriter
from PIL import Image
import os

IMAGE_FORMAT = ['.jpg', '.JPG', '.JPEG', '.jpeg', '.PNG', '.png', '.bmp', '.BMP']

def voc2yolo(voc_file, yolo_file, img_file):
    reader = PascalVocReader(voc_file)
    shapes = reader.getShapes()
    writer = YOLOWriter(
        foldername='',
        filename='',
        imgSize=Image.open(img_file).size[: :-1],

    )
    for s in shapes:
        writer.addBndBox(*s[1], name=s[0], difficult=s[-1])
    writer.save(targetFile=yolo_file)

def yolo2voc(voc_file, yolo_file, img_file):
    reader = YoloReader(yolo_file)
    shapes = reader.getShapes()
    writer = PascalVocWriter(
        foldername='',
        filename='',
        imgSize=Image.open(img_file).size[: :-1],
    )
    for s in shapes:
        writer.addBndBox(*s[1], name=s[0], difficult=s[-1])
    writer.save(targetFile=voc_file)


def main(voc_dir, yolo_dir, img_dir):
    voc_list = os.listdir(voc_dir)
    img_list = os.listdir(img_dir)
    if not os.path.exists(yolo_dir):
        os.mkdir(yolo_dir)
    num = len(voc_list)
    for idx, voc_name in enumerate(voc_list):
        print('{} / {}'.format(idx, num))
        for fm in IMAGE_FORMAT:
            img_name = voc_name.split('.')[0]+fm
            if img_name in img_list:
                voc_file = os.path.join(voc_dir, voc_name)
                yolo_file = os.path.join(yolo_dir, voc_name.split('.')[0]+'.txt')
                img_file = os.path.join(img_dir, img_name)
                voc2yolo(voc_file, yolo_file, img_file)

if __name__ == '__main__':
    # voc2yolo(
    #     voc_file=r'Y:\datasets\lights\train\annotations\traLgt_0001.xml',
    #     yolo_file=r'Y:\datasets\lights\train\labels\traLgt_0001.txt',
    #     img_file=r'Y:\datasets\lights\train\images\traLgt_0001.jpg'
    # )
    main(
        voc_dir=r'G:\dataset\yaogan\aircraft_100\train\annotations',
        yolo_dir=r'G:\dataset\yaogan\aircraft_100\train\labels',
        img_dir=r'G:\dataset\yaogan\aircraft_100\train\images',
    )