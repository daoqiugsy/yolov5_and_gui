from gui.util import parse, write_setting
from gui.ui.setting_window import Ui_SettingWindow
from PyQt5.QtWidgets import QMainWindow, QMessageBox
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QRegExpValidator


class setting_Window(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.max_epoch_edit = None
        self.batch_size = None
        self.ui = Ui_SettingWindow()
        self.ui.setupUi(self)

        self.setting_path = r'setting.yml'
        self.setting_backup_path = r'setting_backup.yml'

        self.setting = parse(self.setting_path)
        self.setting_backup = parse(self.setting_backup_path)
        self.changed_setting = self.setting
       
        self.ui.yingyong_btn.clicked.connect(self.apply)
    def apply(self):
        self.batch_size = self.ui.train_batch_size_edit.text()
        self.max_epoch_edit = self.ui.max_epoch_edit.text()
        
