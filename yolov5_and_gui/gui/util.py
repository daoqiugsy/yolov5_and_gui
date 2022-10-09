import yaml
import numpy as np
import logging as log
from PIL import Image, ImageOps
import os

SUPPORT_FORMAT = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']


def parse(fp):
    with open(fp, 'r', encoding='utf-8') as fd:
        cont = fd.read()
        y = yaml.load(cont, Loader=yaml.FullLoader)
        return y


def write_setting(_dict, fp):
    with open(fp, 'w', encoding='utf-8') as fd:
        yaml.dump(_dict, fd)

#TODO 更改文件夹下图片的分辨率
def resolution(dir: str,save_dir:str,targetResolution):
    '''

    :param dir:
    :return:
    '''
    if targetResolution == 0:
        return
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    target_resolution = targetResolution
    print('转换分辨率为'+str(target_resolution)+'p')
    files = os.listdir(dir)
    files = [os.path.join(dir, name) for name in files]

    for file in files:
        if file.find('.') != -1:
            format = file.split('.')[-1].lower()
        else:
            format = 'dir'
        if format in SUPPORT_FORMAT:


            im = Image.open(file)
            w, h = im.size

            if w < h:
                scale = w / target_resolution
            else:
                scale = h / target_resolution

            im = im.resize((int(w / scale), int(h / scale)))

            im.save(os.path.join(save_dir, os.path.basename(file)))




# modified by mileistone
class Letterbox:
    """ Transform images and annotations to the right network dimensions.

    Args:
        dimension (tuple, optional): Default size for the letterboxing, expressed as a (width, height) tuple; Default **None**
        dataset (lightnet.data.Dataset, optional): Dataset that uses this transform; Default **None**

    Note:
        Create 1 Letterbox object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.
    """
    def __init__(self, dimension=None):
        self.dimension = dimension
        self.dataset = None
        self.pad = None
        self.scale = None
        self.fill_color = 127

    def __call__(self, data):
        if data is None:
            return None
        elif isinstance(data, Image.Image):
            return self._tf_pil(data)
        elif isinstance(data, list):
            return self._tf_boxes(data)
        else:
            log.error(f'Letterbox only works with <PIL images> [{type(data)}]')
            return data

    def _tf_pil(self, img):

        """ Letterbox an image to fit in the network """
        net_w, net_h = self.dimension
        im_w, im_h = img.size
        if im_w == net_w and im_h == net_h:
            self.scale = None
            self.pad = None
            return img
        # Rescaling
        if im_w / net_w >= im_h / net_h:
            self.scale = net_w / im_w
        else:
            self.scale = net_h / im_h
        if self.scale != 1:
            resample_mode = Image.NEAREST #Image.BILINEAR if self.scale > 1 else Image.ANTIALIAS
            img = img.resize((int(self.scale*im_w), int(self.scale*im_h)), resample_mode)
            im_w, im_h = img.size

        if im_w == net_w and im_h == net_h:
            self.pad = None
            return img
        # Padding
        img_np = np.array(img)
        channels = img_np.shape[2] if len(img_np.shape) > 2 else 1
        pad_w = (net_w - im_w) / 2
        pad_h = (net_h - im_h) / 2
        self.pad = (int(pad_w), int(pad_h), int(pad_w+.5), int(pad_h+.5))
        img = ImageOps.expand(img, border=self.pad, fill=(self.fill_color,)*channels)

        return img

    def _tf_boxes(self, boxes):
        for idx in range(len(boxes)):
            if self.pad is not None:
                boxes[idx][0] *= self.scale
                boxes[idx][1] *= self.scale
                boxes[idx][2] *= self.scale
                boxes[idx][3] *= self.scale
                boxes[idx][0] += self.pad[0]
                boxes[idx][1] += self.pad[1]
                boxes[idx][2] += self.pad[0]
                boxes[idx][3] += self.pad[1]

        return boxes





#从文件夹获得图片

# def get_files(dir: str, files: list, root):
#     if os.path.isfile(dir):
#         ff = dir.split('.')[-1]
#         if ff in SUPPORT_FORMAT:
#             files.append(dir.replace(root, '')[1:].replace('/', '\\'))
#
#     elif os.path.isdir(dir):
#         for p in os.listdir(dir):
#             new_dir = os.path.join(dir, p)
#             get_files(new_dir, files, root)

def get_files(dir: str):
    '''

    :param dir:
    :return:
    '''
    files = os.listdir(dir)
    ret = []
    for file in files:
        if file.find('.') != -1:
            format = file.split('.')[-1].lower()
        else:
            format = 'dir'
        if format in SUPPORT_FORMAT:
            ret.append(os.path.join(dir, file))
    return ret



def letterbox(img, net_w, net_h):
    im_w, im_h = img.size

    if im_w / net_w >= im_h / net_h:
        scale = net_w / im_w
    else:
        scale = net_h / im_h
    if scale != 1:
        resample_mode = Image.NEAREST  # Image.BILINEAR if self.scale > 1 else Image.ANTIALIAS
        img = img.resize((int(scale *im_w), int(scale *im_h)), resample_mode)
        im_w, im_h = img.size

    if im_w == net_w and im_h == net_h:
        pad = None
        return img
    # Padding
    img_np = np.array(img)
    channels = img_np.shape[2] if len(img_np.shape) > 2 else 1
    pad_w = (net_w - im_w) / 2
    pad_h = (net_h - im_h) / 2
    pad = (int(pad_w), int(pad_h), int(pad_w +.5), int(pad_h +.5))
    img = ImageOps.expand(img, border=pad, fill=(127, 127, 127))

    return img
