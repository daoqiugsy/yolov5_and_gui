from PyQt5.QtCore import QThread, pyqtSignal
from gui.engine import *
import traceback




class engineThread(QThread):
    finish_signal = pyqtSignal()
    info_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(dict)
    terminal_signal = pyqtSignal()

    def __init__(self, config, mode, display=None):
        super(engineThread, self).__init__()
        self.mode = mode
        hyp = yaml.load(open(r'gui/hyp.yml'))
        if self.mode == 1:
            self.eng = Trainer(
                config=config,
                hyp=hyp,
                info_signal=self.info_signal,
                progress_signal=self.progress_signal,
                plot_view=display,
            )

        elif self.mode == 2:
            self.eng = Tester(
                config=config,
                info_signal=self.info_signal,
                progress_signal=self.progress_signal,
            )
        else:
            self.eng = None

    def _train(self):
        self.eng.run()

    def _test(self):
        self.eng.run()
        self.finish_signal.emit()

    def run(self):
        # train
        if self.mode == 1:
            try:
                self._train()
            except FileNotFoundError as e:  # 我们可以使用except与as+变量名 搭配使用，打印变量名会直接输出报错信息
                self.info_signal.emit('文件夹错误'+str(e))
                traceback.print_exc()
            except Exception as e:  # 我们可以使用except与as+变量名 搭配使用，打印变量名会直接输出报错信息
                self.info_signal.emit('载入模型文件错误'+str(e))
                traceback.print_exc()

            # test
        elif self.mode == 2:
                try:
                    self._test()
                except FileNotFoundError as e:  # 我们可以使用except与as+变量名 搭配使用，打印变量名会直接输出报错信息
                    self.info_signal.emit('文件夹错误' + str(e))
                    traceback.print_exc()
                except Exception as e:  # 我们可以使用except与as+变量名 搭配使用，打印变量名会直接输出报错信息
                    self.info_signal.emit('载入模型文件文件错误' + str(e))
                    traceback.print_exc()

        # error
        elif self.mode == 0:
            self.info_signal.emit('模型文件或文件夹错误')

        self.terminal_signal.emit()

    def quit(self):
        self.eng.quit_flag = 1


def print_info(info):
    print('info: {}'.format(info))


if __name__ == '__main__':
    import yaml

    config = yaml.load(open(r'gui/setting.yml'), Loader=yaml.FullLoader)
    print(config)
    th = engineThread(config, mode=2)
    th.info_signal.connect(print_info)
    th.progress_signal.connect(print_info)
    th.run()
