
from PyQt5.QtWidgets import QWidget, QFileDialog
from gui.ui.help import Ui_help
from data.convert import voc2yolo, yolo2voc
import os

class helpWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.ui = Ui_help()
        self.ui.setupUi(self)






if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    main = helpWidget()
    main.show()
    sys.exit(app.exec_())