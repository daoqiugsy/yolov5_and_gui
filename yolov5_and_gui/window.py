import math
import os
import sys
from pathlib import Path



FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
import val  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.downloads import attempt_download
from utils.general import (LOGGER, NCOLS, check_dataset, check_file, check_git_status, check_img_size,
                           check_requirements, check_suffix, check_yaml, colorstr, get_latest_run, increment_path,
                           init_seeds, intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods,
                           one_cycle, print_args, print_mutation, strip_optimizer, scale_coords)
from utils.loggers import Loggers
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve, plot_labels, plot_one_box2
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, select_device, torch_distributed_zero_first
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
import torch.backends.cudnn as cudnn
from utils.datasets import letterbox
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
import threading
from PyQt5 import QtCore, QtGui, QtWidgets
from gui.format_convert import FormatConvertWidget  # 转换文件
from gui.helpWiget import helpWidget
from PyQt5.QtWidgets import *
from gui.pt2ncnn import Pt2NcnnWidget  # 加密
from gui.ui.yolo_ui_v2 import Ui_YoloWindow
import labelImg
import argparse

import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from tqdm import tqdm
from utils.general import non_max_suppression

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class MainWindow(QMainWindow):
    a = 1

    def __init__(self):
        QMainWindow.__init__(self)

        self.ui = Ui_YoloWindow()
        self.timer_video = QtCore.QTimer()  # 创建定时器
        self.ui.setupUi(self)
        self.cap = cv2.VideoCapture()
        self.step=0
        self.vid_writer = None
        self.num_stop = 1  # 暂停与播放辅助信号，note：通过奇偶来控制暂停与播放
        # self.setting_window = settingWindow()
        self.meihua()
        self.stopEvent = threading.Event()

        argv = []
        self.labelImg_window = labelImg.MainWindow(argv[1] if len(argv) >= 2 else None,
                                                   argv[2] if len(argv) >= 3 else os.path.join(
                                                       os.path.dirname(sys.argv[0]),
                                                       'data', 'predefined_classes.txt'),
                                                   argv[3] if len(argv) >= 4 else None)
        self.stopEvent = threading.Event()
        self.pt2ncnn_window = Pt2NcnnWidget()
        self.format_convert_window = FormatConvertWidget()
        self.help = helpWidget()

        self.ui.action_2.triggered.connect(self.help.show)
        self.ui.actionLabelImg.triggered.connect(self.labelImg_window.show)  # labelimg标注工具
        self.ui.actionxml_txt.triggered.connect(self.format_convert_window.show)  # 标注格式转换工具
        self.ui.actionpt_ncnn.triggered.connect(self.pt2ncnn_window.show)  # pc转pytoc工具
        # self.ui.setting_action.triggered.connect(self.setting_window.show)
        self.ui.pushButton.clicked.connect(self.select_yml_func)
        self.ui.pushButton_2.clicked.connect(self.select_weights_func)
        self.ui.train_stop.clicked.connect(self.train_stop)
        self.ui.test_dir.clicked.connect(self.select_dir)
        self.ui.pushButton_4.clicked.connect(self.select_dir1)
        self.ui.test_weights.clicked.connect(self.open_model)
        self.ui.testmode.clicked.connect(self.model_init)
        self.ui.test_image.clicked.connect(self.button_image_open)
        self.ui.test_video.clicked.connect(self.button_video_open)
        self.ui.test_ca.clicked.connect(self.button_camera_open)
        self.ui.pushButton_3.clicked.connect(self.finish_detect)
        self.ui.train_start.clicked.connect(self.train_1)
        self.ui.pushButton_stop.clicked.connect(self.button_video_stop)
        self.timer_video.timeout.connect(self.show_video_frame)
        self.ui.comboBox.currentIndexChanged[int].connect(self.select)

    def select(self,i):
        self.size=i+1
        print(self.size)

    def meihua(self):
        self.setWindowOpacity(0.9)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

    def select_yml_func(self):  ##选择文件
        self.openyml, _ = QFileDialog.getOpenFileName(self.ui.pushButton, '选择yaml文件',
                                                      'data/')

    def select_weights_func(self):
        self.openfile_preweights, _ = QFileDialog.getOpenFileName(self.ui.pushButton_2, '选择weights文件',

                                                                  'weights/')
    def select_dir(self):
        self.dir=QFileDialog.getExistingDirectory(self.ui.test_dir, "请选择文件夹路径", 'data/')

    def open_model(self):
        self.openfile_name_model, _ = QFileDialog.getOpenFileName(self.ui.test_weights, '选择weights文件',
                                                                  'weights/')
        if not self.openfile_name_model:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"打开权重失败", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            print('加载weights文件地址为：' + str(self.openfile_name_model))

    def model_init(self):
        # 模型相关参数配置
        self.zhixin = self.ui.lineEdit_2.text()
        if type(self.zhixin) is not float:
            self.zhixin=0.4
        self.nms = self.ui.lineEdit_3.text()
        if type(self.nms)is not float:
            self.nms=0.45
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5s.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=float(self.zhixin), help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=float(self.nms), help='IOU threshold for NMS')
        parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        self.opt = parser.parse_args()
        print(self.opt)
        # 默认使用opt中的设置（权重等）来对模型进行初始化
        source, weights, view_img, save_txt, imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size

        # 若openfile_name_model不为空，则使用此权重进行初始化
        if self.openfile_name_model:
            weights = self.openfile_name_model
            print("Using button choose model")


        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        cudnn.benchmark = True

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        print("model initial done")
        # 设置提示框
        QtWidgets.QMessageBox.information(self, u"Notice", u"模型加载完成", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)

    def detect(self, name_list, img):
        '''
        :param name_list: 文件名列表
        :param img: 待检测图片
        :return: info_show:检测输出的文字信息
        '''
        showimg = img
        with torch.no_grad():
            img = letterbox(img, new_shape=self.opt.img_size)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            pred = self.model(img, augment=self.opt.augment)[0]
            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                       agnostic=self.opt.agnostic_nms)
            info_show = ""
            # Process detections
            for i, det in enumerate(pred):
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], showimg.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        label = '%s %.2f' % (self.names[int(cls)], conf)
                        name_list.append(self.names[int(cls)])
                        single_info = plot_one_box2(xyxy, showimg, label=label, color=self.colors[int(cls)],
                                                    line_thickness=2)
                        # print(single_info)
                        info_show = info_show + single_info + "\n"
        return info_show

    def button_image_open(self):
        print('button_image_open')
        name_list = []
        try:
            img_name, _ = QFileDialog.getOpenFileName(self, "打开图片", "data/images",
                                                      "*.jpg;;*.png;;All Files(*)")
        except OSError as reason:
            print('文件打开出错啦！核对路径是否正确' + str(reason))
        else:
            # 判断图片是否为空
            if not img_name:
                QMessageBox.warning(self, u"Warning", u"打开图片失败", buttons=QtWidgets.QMessageBox.Ok,
                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                img = cv2.imread(img_name)
                print("img_name:", img_name)
                info_show = self.detect(name_list, img)
                print(info_show)
                # 获取当前系统时间，作为img文件名
                now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
                file_extension = img_name.split('.')[-1]
                new_filename = now + '.' + file_extension  # 获得文件后缀名
                file_path = self.dir +'/' + new_filename
                print(file_path)
                cv2.imwrite(file_path, img)
                # 检测信息显示在界面

                # 检测结果显示在界面
                self.result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
                self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                          QtGui.QImage.Format_RGB32)
                self.ui.label_3.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
                self.ui.label_3.setScaledContents(True)  # 设置图像自适应界面大小

    def set_video_name_and_path(self):
        # 获取当前系统时间，作为img和video的文件名
        now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        # if vid_cap:  # video
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 视频检测结果存储位置
        save_path = self.dir+'/' + now + '.mp4'

        return fps, w, h, save_path

    def button_video_open(self):
        video_name, _ = QFileDialog.getOpenFileName(self, "打开视频", "data/", "*.mp4;;*.avi;;All Files(*)")
        flag = self.cap.open(video_name)
        if not flag:
            QMessageBox.warning(self, u"Warning", u"打开视频失败", buttons=QtWidgets.QMessageBox.Ok,
                                defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            # -------------------------写入视频----------------------------------#
            fps, w, h, save_path = self.set_video_name_and_path()
            self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

            self.timer_video.start(30)  # 以30ms为间隔，启动或重启定时器
            # 进行视频识别时，关闭其他按键点击功能
            self.ui.test_image.setDisabled(True)
            self.ui.test_video.setDisabled(True)
            self.ui.test_ca.setDisabled(True)

    def button_camera_open(self):
        print("Open camera to detect")
        # 设置使用的摄像头序号，系统自带为0
        camera_num = 0
        # 打开摄像头
        self.cap = cv2.VideoCapture(camera_num)
        # 判断摄像头是否处于打开状态
        bool_open = self.cap.isOpened()
        if not bool_open:
            QMessageBox.warning(self, u"Warning", u"打开摄像头失败", buttons=QtWidgets.QMessageBox.Ok,
                                defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            fps, w, h, save_path = self.set_video_name_and_path()
            fps = 5  # 控制摄像头检测下的fps，Note：保存的视频，播放速度有点快，我只是粗暴的调整了FPS
            self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            self.timer_video.start(30)
            self.ui.test_ca.setDisabled(True)
            self.ui.test_image.setDisabled(True)
            self.ui.test_video.setDisabled(True)

    def show_video_frame(self):
        name_list = []
        flag, img = self.cap.read()
        if img is not None:
            info_show = self.detect(name_list, img)  # 检测结果写入到原始img上
            self.vid_writer.write(img)  # 检测结果写入视频
            print(info_show)
            # 检测信息显示在界面

            show = cv2.resize(img, (640, 480))  # 直接将原始img上的检测结果进行显示
            self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.ui.label_3.setPixmap(QtGui.QPixmap.fromImage(showImage))
            self.ui.label_3.setScaledContents(True)  # 设置图像自适应界面大小

        else:
            self.timer_video.stop()
            # 读写结束，释放资源
            self.cap.release()  # 释放video_capture资源
            self.vid_writer.release()  # 释放video_writer资源
            self.ui.label_3.clear()
            # 视频帧显示期间，禁用其他检测按键功能
            self.ui.test_image.setDisabled(False)
            self.ui.test_video.setDisabled(False)
            self.ui.test_ca.setDisabled(False)

    def button_video_stop(self):
        self.timer_video.blockSignals(False)
        # 暂停检测
        # 若QTimer已经触发，且激活
        if self.timer_video.isActive() == True and self.num_stop % 2 == 1:
            self.ui.pushButton_stop.setText(u'暂停检测')  # 当前状态为暂停状态
            self.num_stop = self.num_stop + 1  # 调整标记信号为偶数
            self.timer_video.blockSignals(True)
        # 继续检测
        else:
            self.num_stop = self.num_stop + 1
            self.ui.pushButton_stop.setText(u'继续检测')

    def finish_detect(self):
        # self.timer_video.stop()
        self.cap.release()  # 释放video_capture资源
        if self.vid_writer:
            self.vid_writer.release()  # 释放video_writer资源
        self.ui.label_3.clear()  # 清空label画布
        # 启动其他检测按键功能
        self.ui.test_ca.setDisabled(False)
        self.ui.test_video.setDisabled(False)
        self.ui.test_image.setDisabled(False)

        # 结束检测时，查看暂停功能是否复位，将暂停功能恢复至初始状态
        # Note:点击暂停之后，num_stop为偶数状态
        if self.num_stop % 2 == 0:
            print("Reset stop/begin!")
            self.ui.pushButton_stop.setText(u'暂停/继续')
            self.num_stop = self.num_stop + 1
            self.timer_video.blockSignals(False)

    def train_1(self):
        self.a = 1
        self.th = threading.Thread(target=self.train_window)
        self.th.start()

    def train_window(self):
        self.printf('正在训练')
        self.cishu= self.ui.lineEdit.text()

        self.opt = self.parse_opt(self.openfile_preweights, self.openyml)
        self.main(self.opt)
        print(self.cishu)

    def printf(self, x):
        self.th2 = threading.Thread(target=self.printf_1(x))
        self.th2.start()

    def printf_1(self, x):
        self.ui.train_textBrowser.append(x)

    def train_stop(self):

        self.a = 0

    def isTure(self):

        return self.a

    def train(self, hyp,  # path/to/hyp.yaml or hyp dictionary
              opt,
              device,
              callbacks
              ):
        save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze, = \
            Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
            opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze

        # Directories
        w = save_dir / 'weights'  # weights dir
        (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
        last, best = w / 'last.pt', w / 'best.pt'

        # Hyperparameters
        if isinstance(hyp, str):
            with open(hyp, errors='ignore') as f:
                hyp = yaml.safe_load(f)  # load hyps dict
        LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

        # Save run settings
        with open(save_dir / 'hyp.yaml', 'w') as f:
            yaml.safe_dump(hyp, f, sort_keys=False)
        with open(save_dir / 'opt.yaml', 'w') as f:
            yaml.safe_dump(vars(opt), f, sort_keys=False)
        data_dict = None

        # Loggers
        if RANK in [-1, 0]:
            loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
            if loggers.wandb:
                data_dict = loggers.wandb.data_dict
                if resume:
                    weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp

            # Register actions
            for k in methods(loggers):
                callbacks.register_action(k, callback=getattr(loggers, k))

        # Config
        plots = not evolve  # create plots
        cuda = device.type != 'cpu'
        init_seeds(1 + RANK)
        with torch_distributed_zero_first(LOCAL_RANK):
            data_dict = data_dict or check_dataset(data)  # check if None
        train_path, val_path = data_dict['train'], data_dict['val']
        nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
        names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
        assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # check
        is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset

        # Model
        check_suffix(weights, '.pt')  # check weights
        pretrained = weights.endswith('.pt')
        if pretrained:
            with torch_distributed_zero_first(LOCAL_RANK):
                weights = attempt_download(weights)  # download if not found locally
            ckpt = torch.load(weights, map_location=device)  # load checkpoint
            model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
            exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
            csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
            model.load_state_dict(csd, strict=False)  # load
            LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
        else:
            model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

        # Freeze
        freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                LOGGER.info(f'freezing {k}')
                v.requires_grad = False

        # Image size
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

        # Batch size
        if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
            batch_size = check_train_batch_size(model, imgsz)

        # Optimizer
        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
        hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
        LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

        g0, g1, g2 = [], [], []  # optimizer parameter groups
        for v in model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
                g2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
                g0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g1.append(v.weight)

        if opt.adam:
            optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
        else:
            optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

        optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
        optimizer.add_param_group({'params': g2})  # add g2 (biases)
        LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                    f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias")
        del g0, g1, g2

        # Scheduler
        if opt.linear_lr:
            lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
        else:
            lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

        # EMA
        ema = ModelEMA(model) if RANK in [-1, 0] else None

        # Resume
        start_epoch, best_fitness = 0, 0.0
        if pretrained:
            # Optimizer
            if ckpt['optimizer'] is not None:
                optimizer.load_state_dict(ckpt['optimizer'])
                best_fitness = ckpt['best_fitness']

            # EMA
            if ema and ckpt.get('ema'):
                ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
                ema.updates = ckpt['updates']

            # Epochs
            start_epoch = ckpt['epoch'] + 1
            if resume:
                assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
            if epochs < start_epoch:
                LOGGER.info(
                    f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
                epochs += ckpt['epoch']  # finetune additional epochs

            del ckpt, csd

        # DP mode
        if cuda and RANK == -1 and torch.cuda.device_count() > 1:
            LOGGER.warning('WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                           'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
            model = torch.nn.DataParallel(model)

        # SyncBatchNorm
        if opt.sync_bn and cuda and RANK != -1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
            LOGGER.info('Using SyncBatchNorm()')

        # Trainloader
        train_loader, dataset = create_dataloader(train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                                  hyp=hyp, augment=True, cache=opt.cache, rect=opt.rect,
                                                  rank=LOCAL_RANK,
                                                  workers=workers, image_weights=opt.image_weights, quad=opt.quad,
                                                  prefix=colorstr('train: '), shuffle=True)
        mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
        nb = len(train_loader)  # number of batches
        assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

        # Process 0
        if RANK in [-1, 0]:
            val_loader = create_dataloader(val_path, imgsz, batch_size // WORLD_SIZE * 2, gs, single_cls,
                                           hyp=hyp, cache=None if noval else opt.cache, rect=True, rank=-1,
                                           workers=workers, pad=0.5,
                                           prefix=colorstr('val: '))[0]

            if not resume:
                labels = np.concatenate(dataset.labels, 0)
                # c = torch.tensor(labels[:, 0])  # classes
                # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
                # model._initialize_biases(cf.to(device))
                if plots:
                    plot_labels(labels, names, save_dir)

                # Anchors
                if not opt.noautoanchor:
                    check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
                model.half().float()  # pre-reduce anchor precision

            callbacks.run('on_pretrain_routine_end')

        # DDP mode
        if cuda and RANK != -1:
            model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

        # Model attributes
        nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
        hyp['box'] *= 3 / nl  # scale to layers
        hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
        hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        hyp['label_smoothing'] = opt.label_smoothing
        model.nc = nc  # attach number of classes to model
        model.hyp = hyp  # attach hyperparameters to model
        model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
        model.names = names

        # Start training
        t0 = time.time()
        nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
        # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
        last_opt_step = -1
        maps = np.zeros(nc)  # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        scheduler.last_epoch = start_epoch - 1  # do not move
        scaler = amp.GradScaler(enabled=cuda)
        stopper = EarlyStopping(patience=opt.patience)
        compute_loss = ComputeLoss(model)  # init loss class
        LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                    f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                    f"Logging results to {colorstr('bold', save_dir)}\n"
                    f'Starting training for {epochs} epochs...')
        for epoch in range(start_epoch,
                           epochs):  # epoch ------------------------------------------------------------------
            model.train()

            # Update image weights (optional, single-GPU only)
            if opt.image_weights:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

            # Update mosaic border (optional)
            # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
            # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

            mloss = torch.zeros(3, device=device)  # mean losses
            if RANK != -1:
                train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(train_loader)
            LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
            if RANK in [-1, 0]:
                pbar = tqdm(pbar, total=nb, ncols=NCOLS, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
            optimizer.zero_grad()
            for i, (
                    imgs, targets, paths,
                    _) in pbar:  # batch -------------------------------------------------------------
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

                # Warmup
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                    accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi,
                                            [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

                # Multi-scale
                if opt.multi_scale:
                    sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / gs) * gs for x in
                              imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Forward
                with amp.autocast(enabled=cuda):
                    pred = model(imgs)  # forward
                    loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                    if RANK != -1:
                        loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                    if opt.quad:
                        loss *= 4.

                # Backward
                scaler.scale(loss).backward()

                # Optimize
                if ni - last_opt_step >= accumulate:
                    scaler.step(optimizer)  # optimizer.step
                    scaler.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)
                    last_opt_step = ni

                # Log
                if RANK in [-1, 0]:
                    mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                    pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                        f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                    callbacks.run('on_train_batch_end', ni, model, imgs, targets, paths, plots, opt.sync_bn)
                # end batch ------------------------------------------------------------------------------------------------
            self.printf(str(epoch) + '/' + str(epochs - 1))
            if self.isTure() == 0:
                self.printf('停止训练')
                return 0
            if epoch == epochs - 1:
                self.printf('训练完成')

            # Scheduler
            lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
            scheduler.step()

            if RANK in [-1, 0]:
                # mAP
                callbacks.run('on_train_epoch_end', epoch=epoch)
                ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
                if not noval or final_epoch:  # Calculate mAP
                    results, maps, _ = val.run(data_dict,
                                               batch_size=batch_size // WORLD_SIZE * 2,
                                               imgsz=imgsz,
                                               model=ema.ema,
                                               single_cls=single_cls,
                                               dataloader=val_loader,
                                               save_dir=save_dir,
                                               plots=False,
                                               callbacks=callbacks,
                                               compute_loss=compute_loss)

                # Update best mAP
                fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                if fi > best_fitness:
                    best_fitness = fi
                log_vals = list(mloss) + list(results) + lr
                callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

                # Save model
                if (not nosave) or (final_epoch and not evolve):  # if save
                    ckpt = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'model': deepcopy(de_parallel(model)).half(),
                            'ema': deepcopy(ema.ema).half(),
                            'updates': ema.updates,
                            'optimizer': optimizer.state_dict(),
                            'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                            'date': datetime.now().isoformat()}

                    # Save last, best and delete
                    torch.save(ckpt, last)
                    if best_fitness == fi:
                        torch.save(ckpt, best)
                    if (epoch > 0) and (opt.save_period > 0) and (epoch % opt.save_period == 0):
                        torch.save(ckpt, w / f'epoch{epoch}.pt')
                    del ckpt
                    callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

                # Stop Single-GPU
                if RANK == -1 and stopper(epoch=epoch, fitness=fi):
                    break

                # Stop DDP TODO: known issues shttps://github.com/ultralytics/yolov5/pull/4576
                # stop = stopper(epoch=epoch, fitness=fi)
                # if RANK == 0:
                #    dist.broadcast_object_list([stop], 0)  # broadcast 'stop' to all ranks

            # Stop DPP
            # with torch_distributed_zero_first(RANK):
            # if stop:
            #    break  # must break all DDP ranks

            # end epoch ----------------------------------------------------------------------------------------------------
        # end training -----------------------------------------------------------------------------------------------------
        if RANK in [-1, 0]:
            LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
            for f in last, best:
                if f.exists():
                    strip_optimizer(f)  # strip optimizers
                    if f is best:
                        LOGGER.info(f'\nValidating {f}...')
                        results, _, _ = val.run(data_dict,
                                                batch_size=batch_size // WORLD_SIZE * 2,
                                                imgsz=imgsz,
                                                model=attempt_load(f, device).half(),
                                                iou_thres=0.65 if is_coco else 0.60,  # best pycocotools results at 0.65
                                                single_cls=single_cls,
                                                dataloader=val_loader,
                                                save_dir=save_dir,
                                                save_json=is_coco,
                                                verbose=True,
                                                plots=True,
                                                callbacks=callbacks,
                                                compute_loss=compute_loss)  # val best model with plots
                        if is_coco:
                            callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)

            callbacks.run('on_train_end', last, best, plots, epoch, results)
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

        torch.cuda.empty_cache()
        return results

    def parse_opt(self, weights=None, yamlx=None, known=False):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', type=str, default=ROOT / weights, help='initial weights path')
        parser.add_argument('--cfg', type=str, default=ROOT / 'yolov5s.yaml', help='model.yaml path')
        parser.add_argument('--data', type=str, default=ROOT / yamlx, help='dataset.yaml path')
        parser.add_argument('--hyp', type=str, default=ROOT / 'hyp.scratch.yaml', help='hyperparameters path')
        parser.add_argument('--epochs', type=int, default=int(self.cishu))
        parser.add_argument('--batch-size', type=int, default=self.size, help='total batch size for all GPUs, -1 for autobatch')
        parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
        parser.add_argument('--rect', action='store_true', help='rectangular training')
        parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
        parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
        parser.add_argument('--noval', action='store_true', help='only validate final epoch')
        parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
        parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
        parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
        parser.add_argument('--cache', type=str, nargs='?', const='ram',
                            help='--cache images in "ram" (default) or "disk"')
        parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
        parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        # parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
        parser.add_argument('--multi-scale', default=True, help='vary img-size +/- 50%%')
        parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
        parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
        parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
        parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
        parser.add_argument('--project', default=ROOT / self.dir1, help='save to project/name')
        parser.add_argument('--name', default='exp', help='save to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--quad', action='store_true', help='quad dataloader')
        parser.add_argument('--linear-lr', action='store_true', help='linear LR')
        parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
        parser.add_argument('--patience', type=int, default=100,
                            help='EarlyStopping patience (epochs without improvement)')
        parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
        parser.add_argument('--save-period', type=int, default=-1,
                            help='Save checkpoint every x epochs (disabled if < 1)')
        parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
        # Weights & Biases arguments
        parser.add_argument('--entity', default=None, help='W&B: Entity')
        parser.add_argument('--upload_dataset', action='store_true', help='W&B: Upload dataset as artifact table')
        parser.add_argument('--bbox_interval', type=int, default=-1,
                            help='W&B: Set bounding-box image logging interval')
        parser.add_argument('--artifact_alias', type=str, default='latest',
                            help='W&B: Version of dataset artifact to use')

        opt = parser.parse_known_args()[0] if known else parser.parse_args()
        return opt

    def main(self, opt, callbacks=Callbacks()):

        # Checks
        if RANK in [-1, 0]:
            print_args(FILE.stem, opt)
            check_git_status()
            check_requirements(exclude=['thop'])

        # Resume
        if opt.resume and not check_wandb_resume(opt) and not opt.evolve:  # resume an interrupted run
            ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
            assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
            with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
                opt = argparse.Namespace(**yaml.safe_load(f))  # replace
            opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
            LOGGER.info(f'Resuming training from {ckpt}')
        else:
            opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
                check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(
                    opt.project)  # checks
            assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
            if opt.evolve:
                opt.project = str(ROOT / 'runs/evolve')
                opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
            opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

        # DDP mode
        device = select_device(opt.device, batch_size=opt.batch_size)
        if LOCAL_RANK != -1:
            assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
            assert opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
            assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
            assert not opt.evolve, '--evolve argument is not compatible with DDP training'
            torch.cuda.set_device(LOCAL_RANK)
            device = torch.device('cuda', LOCAL_RANK)
            dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

        # Train
        if not opt.evolve:
            self.train(opt.hyp, opt, device, callbacks)
            if WORLD_SIZE > 1 and RANK == 0:
                LOGGER.info('Destroying process group... ')
                dist.destroy_process_group()

        # Evolve hyperparameters (optional)
        else:
            # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
            meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                    'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                    'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                    'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                    'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                    'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                    'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                    'box': (1, 0.02, 0.2),  # box loss gain
                    'cls': (1, 0.2, 4.0),  # cls loss gain
                    'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                    'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                    'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                    'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                    'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                    'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                    'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                    'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                    'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                    'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                    'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                    'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                    'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                    'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                    'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                    'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                    'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                    'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                    'mixup': (1, 0.0, 1.0),  # image mixup (probability)
                    'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

            with open(opt.hyp, errors='ignore') as f:
                hyp = yaml.safe_load(f)  # load hyps dict
                if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                    hyp['anchors'] = 3
            opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
            # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
            evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
            if opt.bucket:
                os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {save_dir}')  # download evolve.csv if exists

            for _ in range(opt.evolve):  # generations to evolve
                if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                    # Select parent(s)
                    parent = 'single'  # parent selection method: 'single' or 'weighted'
                    x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                    n = min(5, len(x))  # number of previous results to consider
                    x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                    w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                    if parent == 'single' or len(x) == 1:
                        # x = x[random.randint(0, n - 1)]  # random selection
                        x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                    elif parent == 'weighted':
                        x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                    # Mutate
                    mp, s = 0.8, 0.2  # mutation probability, sigma
                    npr = np.random
                    npr.seed(int(time.time()))
                    g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                    ng = len(meta)
                    v = np.ones(ng)
                    while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                        v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                    for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                        hyp[k] = float(x[i + 7] * v[i])  # mutate

                # Constrain to limits
                for k, v in meta.items():
                    hyp[k] = max(hyp[k], v[1])  # lower limit
                    hyp[k] = min(hyp[k], v[2])  # upper limit
                    hyp[k] = round(hyp[k], 5)  # significant digits

                # Train mutation
                results = self.train(hyp.copy(), opt, device, callbacks)

                # Write mutation results
                print_mutation(results, hyp.copy(), save_dir, opt.bucket)

            # Plot results
            plot_evolve(evolve_csv)
            LOGGER.info(f'Hyperparameter evolution finished\n'
                        f"Results saved to {colorstr('bold', save_dir)}\n"
                        f'Use best hyperparameters example: $ python train.py --hyp {evolve_yaml}')

    def run(self, **kwargs):
        # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
        opt = self.parse_opt(True)
        for k, v in kwargs.items():
            setattr(opt, k, v)
        self.main(opt)

    def print_info(self, text_b):
        def _print_info(text):
            text_b.append(str(text))
            text_b.moveCursor(text_b.textCursor().End)

        return _print_info



    def select_dir1(self):
        self.dir1 = QFileDialog.getExistingDirectory(self.ui.pushButton_4, "请选择文件夹路径", 'data/')

import multiprocessing  # 多进程管理包

if __name__ == '__main__':
    # import warnings
    # warnings.filterwarnings("ignore")
    multiprocessing.freeze_support()
    from QcureUi import cure  # PYQT5

    app = QApplication(sys.argv)
    # main = MainWindow()

    splash = QtWidgets.QSplashScreen(QtGui.QPixmap('gui/test.png'))
    splash.show()
    time.sleep(2)
    # 可以显示启动信息
    splash.showMessage('正在加载……')
    # 关闭启动画面
    splash.close()
    # main.show()
    win = cure.Windows(MainWindow(), '快训检测工具', r'gui/ui/icon.bmp', 'blueDeep', '快训检测工具',
                       r'gui/ui/icon.bmp')
    sys.exit(app.exec_())