from PyQt5.QtWidgets import *
import sys
import winsound
import time

class BeepSound(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('피아노')  # 윈도우 이름과 위치 지정
        self.setGeometry(200, 200, 1200, 100)

        C4 = QPushButton('도', self)  # 버튼 생성
        D4 = QPushButton('레', self)  # 버튼 생성
        E4 = QPushButton('미', self)  # 버튼 생성
        F4 = QPushButton('파', self)  # 버튼 생성
        G4 = QPushButton('솔', self)  # 버튼 생성
        A4 = QPushButton('라', self)  # 버튼 생성
        B4 = QPushButton('시', self)  # 버튼 생성
        C5 = QPushButton('도', self)  # 버튼 생성
        ex= QPushButton('ex) 징글벨', self)  # 버튼 생성
        quitButton = QPushButton('나가기', self)
        self.label = QLabel('주파수 피아노', self)

        C4.setGeometry(10, 10, 100, 30)  # 버튼 위치와 크기 지정
        D4.setGeometry(110, 10, 100, 30)  # 버튼 위치와 크기 지정
        E4.setGeometry(210, 10, 100, 30)  # 버튼 위치와 크기 지정
        F4.setGeometry(310, 10, 100, 30)  # 버튼 위치와 크기 지정
        G4.setGeometry(410, 10, 100, 30)  # 버튼 위치와 크기 지정
        A4.setGeometry(510, 10, 100, 30)  # 버튼 위치와 크기 지정
        B4.setGeometry(610, 10, 100, 30)  # 버튼 위치와 크기 지정
        C5.setGeometry(710, 10, 100, 30)  # 버튼 위치와 크기 지정
        ex.setGeometry(810, 10, 100, 30)  # 버튼 위치와 크기 지정
        quitButton.setGeometry(910, 10, 100, 30)
        self.label.setGeometry(10, 40, 500, 70)

        C4.clicked.connect(self.C4)  # 콜백 함수 지정
        D4.clicked.connect(self.D4)  # 콜백 함수 지정
        E4.clicked.connect(self.E4)  # 콜백 함수 지정
        F4.clicked.connect(self.F4)  # 콜백 함수 지정
        G4.clicked.connect(self.G4)  # 콜백 함수 지정
        A4.clicked.connect(self.A4)  # 콜백 함수 지정
        B4.clicked.connect(self.B4)  # 콜백 함수 지정
        C5.clicked.connect(self.C5)  # 콜백 함수 지정
        ex.clicked.connect(self.ex)  # 콜백 함수 지정
        quitButton.clicked.connect(self.quitFunction)

    def C4(self):
        self.label.setText('주파수 261으로 0.5초 동안 삑 소리를 냅니다.')
        winsound.Beep(261, 500)
    def D4(self):
        self.label.setText('주파수 293으로 0.5초 동안 삑 소리를 냅니다.')
        winsound.Beep(293, 500)
    def E4(self):
        self.label.setText('주파수 329으로 0.5초 동안 삑 소리를 냅니다.')
        winsound.Beep(329, 500)
    def F4(self):
        self.label.setText('주파수 349으로 0.5초 동안 삑 소리를 냅니다.')
        winsound.Beep(349, 500)
    def G4(self):
        self.label.setText('주파수 392으로 0.5초 동안 삑 소리를 냅니다.')
        winsound.Beep(392, 500)
    def A4(self):
        self.label.setText('주파수 440으로 0.5초 동안 삑 소리를 냅니다.')
        winsound.Beep(440, 500)
    def B4(self):
        self.label.setText('주파수 493으로 0.5초 동안 삑 소리를 냅니다.')
        winsound.Beep(493, 500)
    def C5(self):
        self.label.setText('주파수 523으로 0.5초 동안 삑 소리를 냅니다.')
        winsound.Beep(523, 500)
    def ex(self):
        winsound.Beep(440, 300) #라
        winsound.Beep(440, 300) #라
        winsound.Beep(440, 500) #라
        winsound.Beep(440, 300) #라
        winsound.Beep(440, 300) # 라
        winsound.Beep(440, 500) # 라
        winsound.Beep(440, 300) # 라
        winsound.Beep(523, 300) # 도
        winsound.Beep(349, 300) # 파
        winsound.Beep(392, 300) # 솔
        winsound.Beep(440, 500) # 라

    def quitFunction(self):
        self.close()


app = QApplication(sys.argv)
win = BeepSound()
win.show()
app.exec_()