import sys
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt, QPoint, QRect, QSize
from PyQt5.QtGui import QScreen, QCloseEvent
from PyQt5.QtWidgets import QApplication, QLabel, QRubberBand


app = QtWidgets.QApplication(sys.argv)


class MyLabel(QtWidgets.QLabel):

    def __init__(self, parent=None):
        QtWidgets.QLabel.__init__(self, parent)
        self.selection = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
        self.setFocusPolicy(Qt.TabFocus)

    def keyPressEvent(self, qKeyEvent):
        if qKeyEvent.key() == QtCore.Qt.Key_Return:
            print('Enter pressed')
        else:
            super().keyPressEvent(qKeyEvent)

    def mousePressEvent(self, event):
        '''
            Mouse is pressed. If selection is visible either set dragging mode (if close to border) or hide selection.
            If selection is not visible make it visible and start at this point.
        '''

        if event.button() == QtCore.Qt.LeftButton:

            position = QtCore.QPoint(event.pos())
            if self.selection.isVisible():
                # visible selection
                if (self.upper_left - position).manhattanLength() < 20:
                    # close to upper left corner, drag it
                    self.mode = "drag_upper_left"
                elif (self.lower_right - position).manhattanLength() < 20:
                    # close to lower right corner, drag it
                    self.mode = "drag_lower_right"
                else:
                    # clicked somewhere else, hide selection
                    self.selection.hide()
            else:
                # no visible selection, start new selection
                self.upper_left = position
                self.lower_right = position
                self.mode = "drag_lower_right"
                self.selection.show()

    def mouseMoveEvent(self, event):
        '''
            Mouse moved. If selection is visible, drag it according to drag mode.
        '''

        if self.selection.isVisible():
            # visible selection
            if self.mode == "drag_lower_right":
                self.lower_right = QtCore.QPoint(event.pos())
            elif self.mode == "drag_upper_left":
                self.upper_left = QtCore.QPoint(event.pos())
            # update geometry
            self.selection.setGeometry(QtCore.QRect(self.upper_left, self.lower_right).normalized())
            # print(QtCore.QRect(self.upper_left, self.lower_right))

    def mouseReleaseEvent(self, event):
        '''
            Mouse moved. If selection is visible, drag it according to drag mode.
        '''

        if self.selection.isVisible():
            # visible selection
            if self.mode == "drag_lower_right":
                self.lower_right = QtCore.QPoint(event.pos())
            elif self.mode == "drag_upper_left":
                self.upper_left = QtCore.QPoint(event.pos())
            # update geometry
            self.selection.setGeometry(QtCore.QRect(self.upper_left, self.lower_right).normalized())
            # print(QtCore.QRect(self.upper_left, self.lower_right))
            # self.closeEvent(QCloseEvent())
            self.parent().close()


class mainUI(QtWidgets.QWidget):

    def __init__(self):
        super(mainUI, self).__init__()
        self.initUI()

    def initUI(self):
        self.label = MyLabel(self)
        pixmap = QScreen.grabWindow(app.primaryScreen(), app.desktop().winId())

        self.label.setStyleSheet("border: 0px;")
        self.label.setPixmap(pixmap)
        self.label.resize(pixmap.width(), pixmap.height())
        self.label.pixmap().save("grab_01.png", "png")

        geometry = app.desktop().availableGeometry()

        self.setFixedSize(pixmap.width(), pixmap.height())
        topLeftPoint = app.desktop().availableGeometry().topLeft()
        self.move(topLeftPoint)
        # print(self.size())

        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.show()

def rectangle_select():
    window = mainUI()
    app.exec_()
    # print(window.label.upper_left.x())
    x1 = window.label.upper_left.x()
    y1 = window.label.upper_left.y()
    x2 = window.label.lower_right.x()
    y2 = window.label.lower_right.y()
    left = min(x1, x2)
    width = abs(x1 - x2)
    top = min(y1, y2)
    height = abs(y1 - y2)
    # print([left, top, width, height])
    return [left, top, width, height]


if __name__ == '__main__':

    # window = mainUI()
    # app.exec_()
    # print("hello: ", )
    rectangle_select()
    sys.exit()