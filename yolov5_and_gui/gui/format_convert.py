from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QWidget, QFileDialog
from gui.ui.format_convert import Ui_FormatConvertForm
from data.convert import voc2yolo, yolo2voc
import os

class FormatConvertWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.ui = Ui_FormatConvertForm()
        self.ui.setupUi(self)

        self.ui.select_root_pushButton.clicked.connect(self.select_pt_file)
        self.ui.start.clicked.connect(self.start)

    def select_pt_file(self):
        directory_path = QFileDialog.getExistingDirectory(self, "选取文件夹", "./")
        self.ui.root_display_lineEdit.setText(directory_path)

    def start(self):
        task = self.ui.comboBox.currentIndex()
        data_root = self.ui.root_display_lineEdit.text()
        voc_dir = os.path.join(data_root, 'annotations')
        yolo_dir = os.path.join(data_root, 'labels')
        image_dir = os.path.join(data_root, 'images')

        # voc -> yolo
        if task == 0:
            if os.path.exists(yolo_dir):
                self.print_info('根目录下已存在labels文件夹，请删除labels文件夹后重试')
                return
            self.print_info('开始转换')
            try:
                voc2yolo.main(voc_dir, yolo_dir, image_dir)
                self.print_info('转换成功')
            except Exception as e:
                self.print_info('目标文件夹下应包含annotations文件夹\n'+str(e))


        # yolo -> voc
        elif task == 1:
            if os.path.exists(voc_dir):
                self.print_info('根目录下已存在annotations文件夹，请删除annotations文件夹后重试')
                return
            self.print_info('开始转换')
            try:
                yolo2voc.main(voc_dir, yolo_dir, image_dir)
                self.print_info('转换成功')
            except Exception as e:  # 我们可以使用except与as+变量名 搭配使用，打印变量名会直接输出报错信息
                self.print_info('目标文件夹下应包含labels文件夹\n'+str(e))

    def print_info(self, info):
        self.ui.print_textBrowser.append(info)
        self.ui.print_textBrowser.moveCursor(self.ui.print_textBrowser.textCursor().End)


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    main = FormatConvertWidget()
    main.show()
    sys.exit(app.exec_())