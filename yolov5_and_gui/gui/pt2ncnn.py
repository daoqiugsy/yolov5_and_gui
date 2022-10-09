import ctypes
import json
import onnx
import onnxsim
import torch
import os
import torch.nn as nn
# from models.experimental import *
# from utils.datasets import *
# from utils.utils import *
from models.common import Conv
import onnx     # type
#import onnxsim
import argparse
import sys
import traceback
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QWidget, QFileDialog
from gui.ui.pt2ncnn import Ui_pt2ncnn_Form
def encryption(file):
    import ctypes
    pwd = 'chocolate@china.wuhan.wust.2020'
    lib = ctypes.cdll.LoadLibrary('./PytoX.dll')
    exfile = file + "tmp"

    bfile = bytes(file, "utf8")
    bexfile = bytes(exfile, "utf8")
    bpwd = bytes(pwd, "utf8")

#加密
    lib.EncryptFile2File(bfile,bexfile,bpwd)
    os.remove(file)
    os.rename(exfile, file)


def decryption(file):

    import ctypes
    pwd = 'chocolate@china.wuhan.wust.2020'
    lib = ctypes.cdll.LoadLibrary('./PytoX.dll')
    exfile = file + "tmp"

    bfile = bytes(file, "utf8")
    bexfile = bytes(exfile, "utf8")
    bpwd = bytes(pwd, "utf8")
    # 解密
    lib.DecryptFile2File(bfile,bexfile,bpwd)
    os.remove(file)
    os.rename(exfile, file)


class HardSwish(nn.Module):
    def __init__(self):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        return x * (self.relu6(x + 3) / 6)

def fix_act(m):
    if isinstance(m, Conv):
        if m.act.__class__.__name__.find('Hardswish') != -1:
            m.act = HardSwish()

def sim_onnx(onnx_file, save_file):
    parser = argparse.ArgumentParser()
    parser.add_argument('check_n', help='Check whether the output is correct with n random inputs',
                        nargs='?', type=int, default=3)
    parser.add_argument('--enable-fuse-bn', help='Enable ONNX fuse_bn_into_conv optimizer. In some cases it causes incorrect model (https://github.com/onnx/onnx/issues/2677).',
                        action='store_true')
    parser.add_argument('--skip-fuse-bn', help='This argument is deprecated. Fuse-bn has been skippped by default',
                        action='store_true')
    parser.add_argument('--skip-optimization', help='Skip optimization of ONNX optimizers.',
                        action='store_true')
    parser.add_argument(
        '--input-shape', help='The manually-set static input shape, useful when the input shape is dynamic. The value should be "input_name:dim0,dim1,...,dimN" or simply "dim0,dim1,...,dimN" when there is only one input, for example, "data:1,3,224,224" or "1,3,224,224". Note: you might want to use some visualization tools like netron to make sure what the input name and dimension ordering (NCHW or NHWC) is.', type=str, nargs='+')
    args = parser.parse_args()
    print("Simplifying...")
    input_shapes = {}
    if args.input_shape is not None:
        for x in args.input_shape:
            if ':' not in x:
                input_shapes[None] = list(map(int, x.split(',')))
            else:
                pieces = x.split(':')
                # for the input name like input:0
                name, shape = ':'.join(
                    pieces[:-1]), list(map(int, pieces[-1].split(',')))
                input_shapes[name] = shape
    model_opt, check_ok = onnxsim.simplify(
        onnx_file, check_n=args.check_n, perform_optimization=not args.skip_optimization, skip_fuse_bn=not args.enable_fuse_bn, input_shapes=input_shapes)

    onnx.save(model_opt, save_file)

    if check_ok:
        print("Ok!")
    else:
        print("Check failed, please be careful to use the simplified model, or try specifying \"--skip-fuse-bn\" or \"--skip-optimization\" (run \"python3 -m onnxsim -h\" for details)")
        sys.exit(1)

#
# pt_file = r'G:\dataset\yaogan\aircraft_100\train\outputs\final_200.pt'
#
# model = torch.load(pt_file, map_location=torch.device('cpu'))['model'].float()
# model.apply(fix_act)
#
# model.eval()
# model.model[-1].export = True  # set Detect() layer export=True
# # print(model.anchors)
# model.fuse()
# dummy_input1 = torch.randn(1, 3, 640, 640, device='cpu')
# y = model(dummy_input1)
# print([x.size() for x in y])
#
#
# temp_file = 'gui/buffer/temp.onnx'
#
# # torch.onnx.export(model, dummy_input1, temp_file, verbose=True, input_names=input_names, output_names=output_names,
# #                   export_params=True, keep_initializers_as_inputs=True, opset_version=12
# #                   )
#
# torch.onnx.export(model, dummy_input1, temp_file, verbose=False, opset_version=12, input_names=['data'],
#                   output_names=['outputs2', 'outputs1', 'outputs0'])
#
# sim_file = 'gui/buffer/sim.onnx'
#
# sim_onnx(temp_file, sim_file)
#
# save_name = os.path.splitext(pt_file)[0]
#
# cmd = '{} {} {} {}'.format(
#     'onnx2ncnn.exe',
#     sim_file,
#     save_name + '.param',
#     save_name + '.bin',
# )
# res = os.system(cmd)
# print(res)
def to_ncnn(save_file,pt_file, net_w, net_h, info_signal):
    try:
            decryption(pt_file)
            model = torch.load(pt_file, map_location=torch.device('cpu'))['model'].float()
            encryption(pt_file)
            print("载入成功")

            model.apply(fix_act)
            model.eval()
            model.model[-1].export = True
            model.fuse()
    except:
        info_signal.emit(traceback.format_exc())
        return
    # model.cuda()
    dummy_input1 = torch.randn(1, 3, net_w, net_h, device='cpu')

    input_names = ["data"]
    output_names = ["outputs2", "outputs1", "outputs0"]

    temp_file = 'gui/buffer/temp.onnx'

    info_signal.emit('生成模型...')
    torch.onnx.export(model, dummy_input1, temp_file,
                      input_names=input_names, output_names=output_names,
                      verbose=False, opset_version=12,
                      )

    sim_file = 'gui/buffer/sim.onnx'

    info_signal.emit('简化模型...')
    sim_onnx(temp_file, sim_file)
    file_name = os.path.basename(os.path.splitext(pt_file)[0])
    save_name = save_file

    info_signal.emit('生成PyToC模型...')
    cmd = '{} {} {} {}'.format(
        'onnx2ncnn.exe',
        sim_file,
        save_name + '/' + file_name + '.net',
        save_name + '/' + file_name + '.weight',
    )

    res = os.system(cmd)
    if res == 0:
        info_signal.emit('PytoC模型保存至' + save_name + '/' + file_name + '.net')
        info_signal.emit('PytoC模型保存至' + save_name + '/' + file_name + '.weight')
        # TODO: 加密ncnn
        encryption(save_name + '/' + file_name + '.net')
        encryption(save_name + '/' + file_name + '.weight')
        if not os.path.exists(save_name + '/' + 'OSDinitConfig.json'):
            with open('./OSDinitConfig.json', 'r', encoding='utf8')as fp:
                json_data = json.load(fp)

            netName = file_name + '.net'
            weightName = file_name + '.weight'
            json_data['workingDir'] = save_file

            json_data['detect'].append({
                'enable': True,
                'net': netName,
                'weight': weightName
            })

            with open(save_name + '/' + 'OSDinitConfig.json', 'w', encoding='utf8')as fp:
                fp.write(json.dumps(json_data, indent=4))
        else:
            with open(save_name + '/' + 'OSDinitConfig.json', 'r', encoding='utf8')as fp:
                json_data = json.load(fp)

            netName = file_name + '.net'
            weightName = file_name + '.weight'
            json_data['workingDir'] = save_file

            json_data['detect'].append({
                'enable': True,
                'net': netName,
                'weight': weightName
            })

            with open(save_name + '/' + 'OSDinitConfig.json', 'w', encoding='utf8')as fp:
                fp.write(json.dumps(json_data, indent=4))


    else:
        info_signal.emit('生成PyToC模型失败')

    os.remove(sim_file)
    os.remove(temp_file)

def OLDto_ncnn(save_file,pt_file, net_w, net_h, info_signal):


    try:
        lib = ctypes.cdll.LoadLibrary("./PyToX.dll")
        str = pt_file
        file = bytes(str, "utf8")
        file_object = open(str, 'rb')
        chunk = file_object.read(4)
        file_object.close()
        if chunk == b'luga':  # 加密文件头部加的两个字节是st
            print("要解密模型 已加密")

            outFile = ctypes.create_string_buffer(b"jkjkjkj", 2048)

            d = lib.Deciphering(file, outFile)
            print("解密成功")
            outFile = outFile.value.decode()
            model = torch.load(outFile, map_location=torch.device('cpu'))['model'].float()
            print("载入成功")
            if os.path.exists(outFile):  # 如果文件存在
                # 删除文件，可使用以下两种方法。
                os.remove(outFile)
                print('删除成功')
                # os.unlink(path)
            else:
                print('no such file:%s' % outFile)  # 则返回文件不存在
            model.apply(fix_act)
            model.eval()
            model.model[-1].export = True
            model.fuse()


        else:
            print("要解密模型 未加密")
            model = torch.load(pt_file, map_location=torch.device('cpu'))['model'].float()
            print("载入成功")
            lib.Encryption(pt_file)
            model.apply(fix_act)
            model.eval()
            model.model[-1].export = True
            model.fuse()
    except:
        info_signal.emit(traceback.format_exc())
        return
    # model.cuda()
    dummy_input1 = torch.randn(1, 3, net_w, net_h, device='cpu')

    input_names = ["data"]
    output_names = ["outputs2", "outputs1", "outputs0"]

    temp_file = 'gui/buffer/temp.onnx'

    info_signal.emit('生成模型...')
    torch.onnx.export(model, dummy_input1, temp_file,
                      input_names=input_names, output_names=output_names,
                      verbose=False, opset_version=12,
                      )

    sim_file = 'gui/buffer/sim.onnx'

    info_signal.emit('简化模型...')
    sim_onnx(temp_file, sim_file)
    file_name = os.path.basename(os.path.splitext(pt_file)[0])
    save_name = save_file

    info_signal.emit('生成PyToC模型...')
    cmd = '{} {} {} {}'.format(
        'onnx2ncnn.exe',
        sim_file,
        save_name + '/' + file_name + '.net',
        save_name + '/' + file_name + '.weight',
    )

    res = os.system(cmd)
    if res == 0:
        info_signal.emit('PytoC模型保存至' + save_name + '/' + file_name + '.net')
        info_signal.emit('PytoC模型保存至' + save_name + '/' + file_name + '.weight')
    # TODO: 加密ncnn
        encryption(save_name + '/' + file_name + '.net')
        encryption(save_name + '/' + file_name + '.weight')
        if not os.path.exists(save_name + '/' + 'OSDinitConfig.json'):
            with open('./OSDinitConfig.json', 'r', encoding='utf8')as fp:
                json_data = json.load(fp)

            netName = file_name + '.net'
            weightName = file_name + '.weight'
            json_data['workingDir'] = save_file

            json_data['detect'].append({
                'enable': True,
                'net': netName,
                'weight': weightName
            })

            with open(save_name + '/' + 'OSDinitConfig.json', 'w', encoding='utf8')as fp:
                fp.write(json.dumps(json_data, indent=4))
        else:
            with open(save_name + '/' + 'OSDinitConfig.json', 'r', encoding='utf8')as fp:
                json_data = json.load(fp)

            netName = file_name + '.net'
            weightName = file_name + '.weight'
            json_data['workingDir'] = save_file

            json_data['detect'].append({
                'enable': True,
                'net': netName,
                'weight': weightName
            })

            with open(save_name + '/' + 'OSDinitConfig.json', 'w', encoding='utf8')as fp:
                fp.write(json.dumps(json_data, indent=4))


    else:
        info_signal.emit('生成PyToC模型失败')

    os.remove(sim_file)
    os.remove(temp_file)

def ORIto_ncnn(pt_file, net_w, net_h, info_signal):
    try:
        model = torch.load(pt_file, map_location=torch.device('cpu'))['model'].float()
        model.apply(fix_act)
        model.eval()
        model.model[-1].export = True
        model.fuse()
    except:
        info_signal.emit(traceback.format_exc())
        return
    # model.cuda()
    dummy_input1 = torch.randn(1, 3, net_w, net_h, device='cpu')

    input_names = ["data"]
    output_names = ["outputs2", "outputs1", "outputs0"]

    temp_file = 'gui/buffer/temp.onnx'

    info_signal.emit('生成模型...')
    torch.onnx.export(model, dummy_input1, temp_file,
                      input_names=input_names, output_names=output_names,
                      verbose=False, opset_version=12,
                      )

    sim_file = 'gui/buffer/sim.onnx'

    info_signal.emit('简化模型...')
    sim_onnx(temp_file, sim_file)

    save_name = os.path.splitext(pt_file)[0]

    info_signal.emit('生成PyToC模型...')
    cmd = '{} {} {} {}'.format(
        'onnx2ncnn.exe',
        sim_file,
        save_name+'.net',
        save_name+'.weight',
    )



    res = os.system(cmd)
    if res == 0:
        info_signal.emit('PyToC模型保存至' + save_name + '.net')
        info_signal.emit('PyToC模型保存至' + save_name + '.weight')
        encryption(save_name + '.net')
        encryption(save_name + '.weight')



    os.remove(sim_file)
    os.remove(temp_file)



class ToNcnnThread(QThread):
    info_signal = pyqtSignal(str)
    start_signal = pyqtSignal()
    terminal_signal = pyqtSignal()

    def __init__(self,save_file, pt_file, net_w, net_h):
        super(ToNcnnThread, self).__init__()
        self.save_file = save_file
        self.pt_file = pt_file
        self.net_w = net_w
        self.net_h = net_h

    def run(self):
        self.start_signal.emit()
        self.info_signal.emit('正在转换模型...')
        to_ncnn(self.save_file , self.pt_file, self.net_w, self.net_h, self.info_signal)
        self.info_signal.emit('转换结束')
        self.terminal_signal.emit()

class Pt2NcnnWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.ui = Ui_pt2ncnn_Form()
        self.ui.setupUi(self)

        self.ui.select_pt_file.clicked.connect(self.select_pt_file)
        self.ui.save_file.clicked.connect(self.select_save_file)
        self.ui.start.clicked.connect(self.start)
        self.ui.input_h_edit.hide()
        self.ui.input_w_edit.hide()

    def select_pt_file(self):
        openfile_path = QFileDialog.getOpenFileName(self, '选择文件', '', 'weights file (*.pt)')
        openfile_path = openfile_path[0]
        self.ui.file_path_line.setText(openfile_path)

    def select_save_file(self):
        savefile_path = QFileDialog.getExistingDirectory(self, "选取文件夹", "./")
        self.ui.save_path_line_.setText(savefile_path)

    def print_info(self, info):
        self.ui.info_box.append(info)
        self.ui.info_box.moveCursor(self.ui.info_box.textCursor().End)

    def start(self):
        if not os.path.exists(self.ui.file_path_line.text()):
            self.print_info('文件不存在')

        elif not os.path.exists(self.ui.save_path_line_.text()):
            self.print_info('文件夹不存在')
        else:
            net_w = int(self.ui.input_w_edit.text())
            net_h = int(self.ui.input_h_edit.text())
            # 默认网络640
            net_w = 640
            net_h = 640
            pt_file = self.ui.file_path_line.text()
            save_file = self.ui.save_path_line_.text()
            self.T = ToNcnnThread(save_file , pt_file, net_w, net_h)
            self.T.info_signal.connect(self.print_info)
            self.T.start_signal.connect(self.start_enabled(False))
            self.T.terminal_signal.connect(self.start_enabled(True))
            self.T.start()

    def start_enabled(self, _bool):
        def _start_enabled():
            self.ui.start.setEnabled(_bool)
        return _start_enabled

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    main = Pt2NcnnWidget()
    main.show()
    sys.exit(app.exec_())