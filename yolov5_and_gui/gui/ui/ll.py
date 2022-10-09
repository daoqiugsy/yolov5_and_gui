import ctypes
import sys
from PySide6.QtCore import QObject, Signal, QThread
from PySide6.QtWidgets import QMainWindow, QPushButton, QApplication, QTextEdit
from PySide6.QtGui import QTextCursor
from test_code.tmp_train_saxs import train_saxs
import win32con


# 自定义的输出流，将输出重定向到某个地方
class Stream(QObject):
    """Redirects console output to text widget."""
    newText = Signal(str)

    def write(self, text):
        # 发出内容
        self.newText.emit(str(text))

    def flush(self):  # real signature unknown; restored from __doc__
        """ flush(self) """
        pass


# 功能函数执行线程
class MyThread(QThread):  # 线程类
    def __init__(self):
        super(MyThread, self).__init__()
        self.is_on = True

    def run(self):  # 线程执行函数
        self.handle = ctypes.windll.kernel32.OpenThread(  # @UndefinedVariable
            win32con.PROCESS_ALL_ACCESS, False, str(QThread.currentThread()))
        while self.is_on:
            # function code
            train_saxs(data_dir="./data_split_test/saxs/train", epochs=5)
            self.is_on = False  # 训练完成后，终止线程的执行


# GUI代码
class GenMast(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

        # Custom output stream.
        self.sm = Stream()
        self.sm.newText.connect(self.onUpdateText)
        sys.stdout = self.sm
        sys.stderr = self.sm

        self.my_thread = MyThread()
        # 拓展：线程结束后执行的操作
        self.my_thread.finished.connect(self.thread_finish_process)

    def onUpdateText(self, text):
        """Write console output to text widget."""
        cursor = self.process.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.process.setTextCursor(cursor)
        self.process.ensureCursorVisible()

    def closeEvent(self, event):
        # Return stdout to defaults.
        sys.stdout = sys.__stdout__
        super().closeEvent(event)

    def initUI(self):
        """Creates UI window on launch."""
        # Button for generating the master list.
        btnGenMast = QPushButton('Run', self)
        btnGenMast.move(450, 50)
        btnGenMast.resize(100, 200)
        btnGenMast.clicked.connect(self.genMastClicked)

        # Create the text output widget.
        self.process = QTextEdit(self, readOnly=True)
        self.process.ensureCursorVisible()
        self.process.setLineWrapColumnOrWidth(500)
        self.process.setLineWrapMode(QTextEdit.FixedPixelWidth)
        self.process.setFixedWidth(400)
        self.process.setFixedHeight(200)
        self.process.move(30, 50)

        # Set window size and title, then show the window.
        self.setGeometry(300, 300, 600, 300)
        self.setWindowTitle('Generate Master')
        self.show()

    def train_and_echo_to_gui(self):
        self.my_thread.start()

        def thread_finish_process(self):
            print("thread is end...")

    def genMastClicked(self):
        """Runs the main function."""
        print('Begin.')
        self.train_and_echo_to_gui()
        print('Done.')


if __name__ == '__main__':
    # Run the application.
    app = QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    gui = GenMast()
    sys.exit(app.exec_())
