# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'format_convert.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_FormatConvertForm(object):
    def setupUi(self, FormatConvertForm):
        FormatConvertForm.setObjectName("FormatConvertForm")
        FormatConvertForm.resize(748, 312)
        self.gridLayout = QtWidgets.QGridLayout(FormatConvertForm)
        self.gridLayout.setObjectName("gridLayout")
        self.comboBox = QtWidgets.QComboBox(FormatConvertForm)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.gridLayout.addWidget(self.comboBox, 0, 1, 1, 1)
        self.select_root_pushButton = QtWidgets.QPushButton(FormatConvertForm)
        self.select_root_pushButton.setObjectName("select_root_pushButton")
        self.gridLayout.addWidget(self.select_root_pushButton, 1, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(FormatConvertForm)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.label = QtWidgets.QLabel(FormatConvertForm)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.print_textBrowser = QtWidgets.QTextBrowser(FormatConvertForm)
        self.print_textBrowser.setObjectName("print_textBrowser")
        self.gridLayout.addWidget(self.print_textBrowser, 3, 0, 1, 3)
        self.root_display_lineEdit = QtWidgets.QLineEdit(FormatConvertForm)
        self.root_display_lineEdit.setObjectName("root_display_lineEdit")
        self.gridLayout.addWidget(self.root_display_lineEdit, 1, 1, 1, 1)
        self.start = QtWidgets.QPushButton(FormatConvertForm)
        self.start.setObjectName("start")
        self.gridLayout.addWidget(self.start, 0, 2, 1, 1)

        self.retranslateUi(FormatConvertForm)
        QtCore.QMetaObject.connectSlotsByName(FormatConvertForm)

    def retranslateUi(self, FormatConvertForm):
        _translate = QtCore.QCoreApplication.translate
        FormatConvertForm.setWindowTitle(_translate("FormatConvertForm", "标注格式转换工具"))
        self.comboBox.setItemText(0, _translate("FormatConvertForm", "xml --> txt"))
        self.comboBox.setItemText(1, _translate("FormatConvertForm", "txt  --> xml"))
        self.select_root_pushButton.setText(_translate("FormatConvertForm", "..."))
        self.label_2.setText(_translate("FormatConvertForm", "选择功能"))
        self.label.setText(_translate("FormatConvertForm", "选择根目录"))
        self.start.setText(_translate("FormatConvertForm", "开始"))

