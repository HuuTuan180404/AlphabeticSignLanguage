import cv2
import time
import pickle
import numpy as np
import mediapipe as mp
from PyQt6 import QtCore, QtGui, QtWidgets

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class CameraThread(QtCore.QThread):
    change_frame = QtCore.pyqtSignal(QtGui.QImage)
    update_text = QtCore.pyqtSignal(str)
    import os
    os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

    def run(self):
        hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7)
        model = pickle.load(open('../MLPClassifier/MLP_model.p', 'rb'))

        cap = cv2.VideoCapture(0)
        pre_class = "-"
        # current_class = ""
        stable_start_time = None
        stable_duration = 1
        KQ = ""
        is_stable = False

        while True:
            x_, y_ = [], []
            ret, frame = cap.read()
            if not ret:
                break
            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    data_aux = []
                    for lm in hand_landmarks.landmark:
                        data_aux.append(lm.x)
                        data_aux.append(lm.y)
                        x_.append(lm.x)
                        y_.append(lm.y)

                x1 = int(min(x_) * W)
                y1 = int(min(y_) * H)

                prediction = model.predict([np.array(data_aux)])
                current_class = prediction[0]

                if current_class != pre_class:
                    stable_start_time = time.time()
                    pre_class = current_class
                    is_stable = False
                else:
                    if not is_stable and time.time() - stable_start_time > stable_duration:
                        if current_class == 'space':
                            KQ += '_'
                        elif current_class == 'del':
                            KQ = KQ[:len(KQ)-1]
                        else:
                            KQ += current_class
                        is_stable = True

                cv2.putText(frame, prediction[0], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

            # Convert to QImage and emit to GUI
            qt_img = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0],
                                  frame.strides[0], QtGui.QImage.Format.Format_BGR888)
            self.change_frame.emit(qt_img)
            self.update_text.emit(KQ)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(868, 622)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.txtResult = QtWidgets.QTextEdit(parent=self.centralwidget)
        self.txtResult.setGeometry(QtCore.QRect(440, 10, 401, 71))
        self.txtResult.setObjectName("txtResult")

        self.label_3 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(440, 90, 411, 421))
        self.label_3.setPixmap(QtGui.QPixmap("f.png"))
        self.label_3.setScaledContents(True)
        self.label_3.setObjectName("label_3")

        self.frame = QtWidgets.QLabel(parent=self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(40, 20, 381, 491))
        self.frame.setScaledContents(True)
        self.frame.setObjectName("frame")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Khởi chạy camera thread
        self.thread = CameraThread()
        self.thread.change_frame.connect(self.update_frame)
        self.thread.update_text.connect(self.update_textbox)
        self.thread.start()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Hand Detection App"))

    def update_frame(self, image):
        self.frame.setPixmap(QtGui.QPixmap.fromImage(image))

    def update_textbox(self, text):
        self.txtResult.setText(text)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
