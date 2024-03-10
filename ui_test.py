from PyQt5 import QtCore,QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer
from kenshutsu import *
import cv2
import sys, os, xlwt
import numpy as np
import time






class Ui_MainWindow(object):
    def __init__(self):
        self.video_control = False
        self.RowLength = 0
        self.Data = [['文件名称', '录入时间', '车牌号码', '车牌类型', '识别耗时', '车牌信息']]
        self.data_base = []
        self.Get_Picture_Information = Get_Information()



    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1213, 670)
        MainWindow.setFixedSize(1213, 670)  # 设置窗体固定大小
        MainWindow.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        MainWindow.setWindowIcon(QtGui.QIcon('./ui_icon.jpg'))
        # MainWindow.setStyleSheet("background-color: rgba(214, 213, 183, 0.8);")




        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(690, 10, 511, 491))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 509, 489))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.label_0 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_0.setGeometry(QtCore.QRect(10, 10, 111, 20))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_0.setFont(font)
        self.label_0.setObjectName("label_0")
        self.label = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label.setGeometry(QtCore.QRect(10, 40, 481, 441))
        self.label.setObjectName("label")
        self.label.setAlignment(Qt.AlignCenter)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.scrollArea_2 = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea_2.setGeometry(QtCore.QRect(10, 10, 671, 631))
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollArea_2.setObjectName("scrollArea_2")

        self.scrollAreaWidgetContents_1 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_1.setGeometry(QtCore.QRect(0, 0, 669, 629))
        self.scrollAreaWidgetContents_1.setObjectName("scrollAreaWidgetContents_1")

        background_image_path = "./ui_icon.jpg"
        stylesheet = stylesheet = """
QWidget#scrollAreaWidgetContents_1 {
    background-image: url('""" + background_image_path + """');
    background-repeat: no-repeat;
    background-position: center;
}
"""
        self.scrollAreaWidgetContents_1.setStyleSheet(stylesheet)

        self.label_1 = QtWidgets.QLabel(self.scrollAreaWidgetContents_1)
        self.label_1.setGeometry(QtCore.QRect(10, 10, 151, 20))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_1.setFont(font)
        self.label_1.setObjectName("label_1")
        
        self.tableWidget = QtWidgets.QTableWidget(self.scrollAreaWidgetContents_1)
        self.tableWidget.setGeometry(QtCore.QRect(8, 40, 655, 585))  # 581))
        self.tableWidget.setObjectName("tableWidget")

        self.tableWidget.setStyleSheet("""
    QTableWidget {
        background-color: rgba(255, 255, 255, 190);
    }
""")

        self.tableWidget.setColumnCount(4)
        self.tableWidget.setColumnWidth(0, 185)  # 设置1列的宽度
        self.tableWidget.setColumnWidth(1, 170)  # 设置2列的宽度
        self.tableWidget.setColumnWidth(2, 155)  # 设置3列的宽度
        self.tableWidget.setColumnWidth(3, 175)  # 设置4列的宽度

        self.tableWidget.setHorizontalHeaderLabels(["车牌号码", "车牌类型", "识别耗时", "置信度"])
        self.tableWidget.setRowCount(self.RowLength)
        self.tableWidget.verticalHeader().setVisible(False)  # 隐藏垂直表头)
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableWidget.raise_()
        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_1)
        self.scrollArea_3 = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea_3.setGeometry(QtCore.QRect(690, 510, 341, 131))
        self.scrollArea_3.setWidgetResizable(True)
        self.scrollArea_3.setObjectName("scrollArea_3")
        self.scrollAreaWidgetContents_3 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_3.setGeometry(QtCore.QRect(0, 0, 339, 129))
        self.scrollAreaWidgetContents_3.setObjectName("scrollAreaWidgetContents_3")
        self.label_2 = QtWidgets.QLabel(self.scrollAreaWidgetContents_3)
        self.label_2.setGeometry(QtCore.QRect(10, 10, 321, 100))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.scrollArea_3.setWidget(self.scrollAreaWidgetContents_3)
        self.scrollArea_4 = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea_4.setGeometry(QtCore.QRect(1040, 510, 161, 131))
        self.scrollArea_4.setWidgetResizable(True)
        self.scrollArea_4.setObjectName("scrollArea_4")
        self.scrollAreaWidgetContents_4 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_4.setGeometry(QtCore.QRect(0, 0, 159, 129))
        self.scrollAreaWidgetContents_4.setObjectName("scrollAreaWidgetContents_4")
        self.pushButton_2 = QtWidgets.QPushButton(self.scrollAreaWidgetContents_4)
        self.pushButton_2.setGeometry(QtCore.QRect(20, 50, 121, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton = QtWidgets.QPushButton(self.scrollAreaWidgetContents_4)
        self.pushButton.setGeometry(QtCore.QRect(20, 90, 121, 31))
        self.pushButton.setObjectName("pushButton")
        self.label_4 = QtWidgets.QLabel(self.scrollAreaWidgetContents_4)
        self.label_4.setGeometry(QtCore.QRect(10, 10, 111, 20))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.scrollArea_4.setWidget(self.scrollAreaWidgetContents_4)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.pushButton.clicked.connect(self.__openimage)  # 设置点击事件
        self.pushButton_2.clicked.connect(self.__openvideo)  # 设置点击事件
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.ProjectPath = os.getcwd()  # 获取当前工程文件位置

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "License plate Detection"))
        self.label_0.setText(_translate("MainWindow", "识别区域："))
        self.label.setText(_translate("MainWindow", ""))
        self.label_1.setText(_translate("MainWindow", "识别结果记录："))
        self.label_2.setText(_translate("MainWindow", "萍水相逢，尽是他乡之客"))
        self.pushButton.setText(_translate("MainWindow", "图片识别"))
        self.pushButton_2.setText(_translate("MainWindow", "视频识别"))
        self.label_4.setText(_translate("MainWindow", "导入："))
        self.scrollAreaWidgetContents_1.show()

    def __show(self, names, thes, colours, times):

        for i in range(len(names)):
            if names[i] in self.data_base:
                continue
            else:
                self.data_base.append(names[i])
            # 显示表格
            self.RowLength = self.RowLength + 1
            if self.RowLength > 18:
                self.tableWidget.setColumnWidth(15, 157)
            self.tableWidget.setRowCount(self.RowLength)
            self.tableWidget.setItem(self.RowLength - 1, 0, QTableWidgetItem(names[i]))
            self.tableWidget.setItem(self.RowLength - 1, 1, QTableWidgetItem(colours[i]))
            self.tableWidget.setItem(self.RowLength - 1, 2, QTableWidgetItem(times[:4]))
            self.tableWidget.setItem(self.RowLength - 1, 3, QTableWidgetItem(str(thes[i])[:4]))

    def __openvideo(self):
        self.video_control = False
        path, filetype = QFileDialog.getOpenFileName(None, "选择文件", self.ProjectPath,
                                                     "MP4 Video (*.mp4);;MKV Video (*.mkv)")
        
        if path == "":  # 未选择文件
            return
        self.video_control = True
        
        cap = cv2.VideoCapture(path) # 0 代表摄像头，或者替换为视频文件路径

        frame_count = 0

        while cap.isOpened() and self.video_control:
            ret, frame = cap.read()
            if not ret:
                break

            # 每隔三帧处理一帧
            if frame_count % 3 == 0:
                time.sleep(0.01)    #视频的播放速度取决于这段代码(识别)的处理速度，故只能手动添加延时，否则会有快进效果(一秒内输出过多帧图像)
                _start = time.time()
                result = self.Get_Picture_Information(frame)
                if not result[1]:  #如果没有识别到车牌则下一帧
                    self.stream_play(frame)
                else:
                    self.stream_play(result[0], default=True)
                    self.__show(result[1], result[2], result[3], str(time.time()-_start))

            frame_count += 1

            keyvalue = cv2.waitKey(1)

            # 按 'q' 键退出
            if keyvalue & 0xFF == ord('q'):
                break
            # 空格暂停
            elif keyvalue & 0xFF == ord(' '):
                cv2.waitKey(0)

    def __openimage(self):
        self.video_control = False
        path, filetype = QFileDialog.getOpenFileName(None, "选择文件", self.ProjectPath,
                                                     "JPEG Image (*.jpg);;PNG Image (*.png)")

        if path == "":  # 未选择文件
            return
        filename = path.split('/')[-1]

        cv_data = cv2.imread(path)
        _start = time.time()
        result = self.Get_Picture_Information(cv_data)

        self.stream_play(cv_data=result[0])

        if result[1] is not None:
            self.__show(result[1], result[2], result[3], str(time.time()-_start))
        else:
            QMessageBox.warning(None, "Error", "无法识别此图像！", QMessageBox.Yes)


    # 手搓视频流处理(逐帧)
    def stream_play(self, cv_data, default=False):

        size = cv_data.shape
        cv_data = cv2.cvtColor(cv_data, cv2.COLOR_BGR2RGB)
        cv_data = QImage(cv_data.data, size[1], size[0], size[1] * size[2], QImage.Format_RGB888)
        if size[0] / size[1] > 1.0907:
            w = size[1] * self.label.height() / size[0]
            h = self.label.height()
            # if default:
            #     jpg = QtGui.QPixmap(cv_data).scaled(int(w), int(h))
            # else:
            #     # jpg = QtGui.QPixmap(cv_data).scaled(w, h)
            jpg = QtGui.QPixmap(cv_data).scaled(int(w), int(h))
        elif size[0] / size[1] < 1.0907:
            w = self.label.width()
            h = size[0] * self.label.width() / size[1]
            # if default:
            #     jpg = QtGui.QPixmap(cv_data).scaled(int(w), int(h))
            # else:
            #     # jpg = QtGui.QPixmap(cv_data).scaled(w, h)
            jpg = QtGui.QPixmap(cv_data).scaled(int(w), int(h))
        else:
            jpg = QtGui.QPixmap(cv_data).scaled(self.label.width(), self.label.height())


        self.label.setPixmap(jpg)




# 重写MainWindow类
class MainWindow(QtWidgets.QMainWindow):

    def closeEvent(self, event):
        # ui.video_control = False
        reply = QtWidgets.QMessageBox.question(self, '提示',
                                               "是否要退出程序？",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            ui.video_control = False
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
