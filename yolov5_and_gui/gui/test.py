# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
import sys
import os
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication
import labelImg


class Ui_MainWindow(object):
    def label(self):
          while 1:
              labelImg.minn()
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1031, 865)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(0, 0, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(90, 0, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(180, 0, 93, 28))
        self.pushButton_3.setObjectName("pushButton_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action = QtWidgets.QAction(MainWindow)
        self.action.setChecked(False)
        self.action.setObjectName("action")
        self.action_2 = QtWidgets.QAction(MainWindow)
        self.action_2.setCheckable(True)
        self.action_2.setChecked(True)
        self.action_2.setObjectName("action_2")
        self.action_3 = QtWidgets.QAction(MainWindow)
        self.action_3.setObjectName("action_3")
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        argv = []
        self.labelImg_window = labelImg.MainWindow(argv[1] if len(argv) >= 2 else None,
                                                   argv[2] if len(argv) >= 3 else os.path.join(
                                                       os.path.dirname(sys.argv[0]),
                                                       'data', 'predefined_classes.txt'),
                                                   argv[3] if len(argv) >= 4 else None)


        self.pushButton.clicked.connect(lambda :self.label())

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "yolov5"))
        self.pushButton.setText(_translate("MainWindow", "labelImg"))
        self.pushButton_2.setText(_translate("MainWindow", "训练数据"))
        self.pushButton_3.setText(_translate("MainWindow", "测试数据"))
        self.action.setText(_translate("MainWindow", "数据集标注"))
        self.action_2.setText(_translate("MainWindow", "测试模型"))
        self.action_3.setText(_translate("MainWindow", "训练模型"))
if __name__ == '__main__':
    app = QApplication(sys.argv)

    MainWindow=QtWidgets.QMainWindow()
    UI=Ui_MainWindow()
    UI.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())