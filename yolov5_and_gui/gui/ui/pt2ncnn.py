# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'pt2ncnn.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_pt2ncnn_Form(object):
    def setupUi(self, pt2ncnn_Form):
        pt2ncnn_Form.setObjectName("pt2ncnn_Form")
        pt2ncnn_Form.resize(581, 383)
        self.gridLayout_2 = QtWidgets.QGridLayout(pt2ncnn_Form)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(pt2ncnn_Form)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 2, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(pt2ncnn_Form)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.info_box = QtWidgets.QTextBrowser(pt2ncnn_Form)
        self.info_box.setObjectName("info_box")
        self.gridLayout.addWidget(self.info_box, 5, 0, 1, 7)
        self.input_w_edit = QtWidgets.QLineEdit(pt2ncnn_Form)
        self.input_w_edit.setObjectName("input_w_edit")
        self.gridLayout.addWidget(self.input_w_edit, 6, 2, 1, 1)
        self.select_pt_file = QtWidgets.QPushButton(pt2ncnn_Form)
        self.select_pt_file.setObjectName("select_pt_file")
        self.gridLayout.addWidget(self.select_pt_file, 0, 5, 1, 2)
        self.file_path_line = QtWidgets.QLineEdit(pt2ncnn_Form)
        self.file_path_line.setText("")
        self.file_path_line.setReadOnly(True)
        self.file_path_line.setObjectName("file_path_line")
        self.gridLayout.addWidget(self.file_path_line, 0, 1, 1, 4)
        self.input_h_edit = QtWidgets.QLineEdit(pt2ncnn_Form)
        self.input_h_edit.setObjectName("input_h_edit")
        self.gridLayout.addWidget(self.input_h_edit, 6, 4, 1, 1)
        self.start = QtWidgets.QPushButton(pt2ncnn_Form)
        self.start.setObjectName("start")
        self.gridLayout.addWidget(self.start, 6, 0, 1, 1)
        self.save_path_line_ = QtWidgets.QLineEdit(pt2ncnn_Form)
        self.save_path_line_.setObjectName("save_path_line_")
        self.gridLayout.addWidget(self.save_path_line_, 2, 1, 1, 4)
        self.save_file = QtWidgets.QPushButton(pt2ncnn_Form)
        self.save_file.setObjectName("save_file")
        self.gridLayout.addWidget(self.save_file, 2, 5, 1, 2)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 2, 1, 1)

        self.retranslateUi(pt2ncnn_Form)
        QtCore.QMetaObject.connectSlotsByName(pt2ncnn_Form)

    def retranslateUi(self, pt2ncnn_Form):
        _translate = QtCore.QCoreApplication.translate
        pt2ncnn_Form.setWindowTitle(_translate("pt2ncnn_Form", "pt转PytoC"))
        self.label.setText(_translate("pt2ncnn_Form", "选择保存路径"))
        self.label_2.setText(_translate("pt2ncnn_Form", "选择det模型文件"))
        self.input_w_edit.setText(_translate("pt2ncnn_Form", "640"))
        self.select_pt_file.setText(_translate("pt2ncnn_Form", "..."))
        self.input_h_edit.setText(_translate("pt2ncnn_Form", "640"))
        self.start.setText(_translate("pt2ncnn_Form", "开始"))
        self.save_file.setText(_translate("pt2ncnn_Form", "..."))