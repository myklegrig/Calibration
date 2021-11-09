import sys  # sys нужен для передачи argv в QApplication
import cv2 as cv
import glob
from matplotlib import pyplot
from PyQt5 import QtGui, QtCore, QtWidgets
import Interface
from PyQt5.QtWidgets import * 
from PyQt5.QtGui import * 
from PyQt5.QtCore import * 
import numpy as np
from PIL import Image
import os
import time
from datetime import datetime
from threading import Thread
from PyQt5.QtCore import QThread,  pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PySide2.QtGui import QIcon, QFont
import yaml
from detecto import core, utils


cap = cv.VideoCapture(2, cv.CAP_V4L2)

cap.set(cv.CAP_PROP_FRAME_WIDTH, 3448)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 808)

def decode(frame):
    left = np.zeros((800,1264,3), np.uint8)
    right = np.zeros((800,1264,3), np.uint8)
    
    for i in range(800):
        left[i] = frame[i, 64: 1280 + 48] 
        right[i] = frame[i, 1280 + 48: 1280 + 48 + 1264] 
    
    return (left, right)


class Model:
    def __init__(self):
        self.model = core.Model.load('model_weights.pth', ['chessboard'])

    def get_box(self, frame):
        _, box, _ = self.model.predict(frame)
        return box




class Thread1(QThread):
    changePixmap1 = pyqtSignal(QImage)
    changePixmap2 = pyqtSignal(QImage)
   
   
    def __init__(self, *args, **kwargs):
        super().__init__()
        # self.model_l = Model()
        # self.model_r = Model()
        # self.boxes_l = []
        # self.boxes_r = []

    def run(self):
        
        while cap.isOpened():
            if AppInterface.fl == False:
                ret, frame = cap.read()   
                left, right = decode(frame) 

                if ret:
                    # box_l = self.model_l.get_box(left)
                    # box_r = self.model_r.get_box(right)
                    # print('p1')
                    
                    # if box_l.numpy().any() and box_r.numpy().any():
                    #     self.boxes_l.append([int(box_l[0][0]), int(box_l[0][1]), int(box_l[0][2]), int(box_l[0][3])])
                    #     self.boxes_r.append([int(box_r[0][0]), int(box_r[0][1]), int(box_r[0][2]), int(box_r[0][3])])
                    # print('p2')

                    # if self.boxes_l and self.boxes_r:
                    #     print('p3')
                    #     for box_l in self.boxes_l:
                    #         print('before')

                    #         left = cv.rectangle(left, (box_l[0], box_l[1]), (box_l[2], box_l[3]), (0, 255, 0), -1)
                    #         #right = cv.rectangle(right, (box_r[0], box_r[1]), (box_r[2], box_r[3]), (0, 255, 0), -1)
                    #         print('after')

                    im1 = cv.cvtColor(left, cv.COLOR_BGR2RGB)
                    im2 = cv.cvtColor(right, cv.COLOR_BGR2RGB)
                    height1, width1, channel1 = im1.shape
                    height2, width2, channel2 = im2.shape
                    step1 = channel1 * width1
                    step2 = channel2 * width2
                    qImg1 = QImage(im1.data, width1, height1, step1, QImage.Format_RGB888)
                    qImg2 = QImage(im2.data, width2, height2, step2, QImage.Format_RGB888)
                    
                    self.changePixmap1.emit(qImg1)
                    self.changePixmap2.emit(qImg2)
            else:
                print('exit')
                break




class Calibration(QThread):

    changePixmap1L = pyqtSignal(QImage)
    changePixmap2R = pyqtSignal(QImage)
   
    def __init__(self, calwindow, parent=None):
        super().__init__()
        self.calwindow = calwindow

    def calibration(calwindow):

        chessboardSize = (7,7)
        frameSize = (640,480)



        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)



        objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

        objp = objp * 20
        print(objp)


        objpoints = [] # 3D точки реального пространства 
        imgpL = [] # 2D точки в плоскости левого изображения
        imgpR = [] # 2D точки в плоскости правого изображения


        imagesLeft = glob.glob('/home/mikhail/Work/final/images/imageLeft/*.png')
        imagesRight = glob.glob('/home/mikhail/Work/final/images/imageRight/*.png')
        print('success')
        y=0
        for imgLeft, imgRight in zip(imagesLeft, imagesRight):
            y+=1
            print(y)
            imgL = cv.imread(imgLeft)
            imgR = cv.imread(imgRight)
            grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
            grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
            print('success')
            # Нахождение углов шахматной доски
            retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
            retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)
            print(retL)
            print(retR)
            #Если найдено успешно, добавление точек объектов и изображений
            if retL and retR == True:
                
                print('success')
                objpoints.append(objp)

                cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
                imgpL.append(cornersL)

                cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
                imgpR.append(cornersR)

                # Нарисовать и отобразить углы
                cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
                #cv.imshow('img left', imgL)
                cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)    
                #cv.imshow('img right', imgR)
                im1 = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
                im2 = cv.cvtColor(imgR, cv.COLOR_BGR2RGB)
                height1, width1, channel1 = im1.shape
                height2, width2, channel2 = im2.shape
                step1 = channel1 * width1
                step2 = channel2 * width2
                qImg1 = QImage(im1.data, width1, height1, step1, QImage.Format_RGB888)
                qImg2 = QImage(im2.data, width2, height2, step2, QImage.Format_RGB888)
                calwindow.changePixmap1L.emit(qImg1)
                calwindow.changePixmap2R.emit(qImg2)
                time.sleep(6)


            
                #cv.destroyAllWindows()


        cv.destroyAllWindows()




        #КАЛИБРОВКА КАЖДОЙ КАМЕРЫ В ОТДЕЛЬНОСТИ

        retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpL, frameSize, None, None)
        heightL, widthL, channelsL = imgL.shape
        newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

        retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpR, frameSize, None, None)
        heightR, widthR, channelsR = imgR.shape
        newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))



        #КАЛИБРОВКА СТЕРЕОПАРЫ

        flags = 0
        flags |= cv.CALIB_FIX_INTRINSIC


        criteria_stereo= (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


        retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpL, imgpR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)

        # transform the matrix and distortion coefficients to writable lists
        data = {'camera_matrix': np.asarray(fundamentalMatrix).tolist(),
        'dist_coeff_left': np.asarray(distL).tolist(),
        'dist_coeff_right': np.asarray(distR).tolist(),
        'camera_matrix_L': np.asarray(newCameraMatrixL).tolist(),
        'camera_matrix_R': np.asarray(newCameraMatrixR).tolist(),
        'R_matrix': np.asarray(rot).tolist(),
        'T_matrix': np.asarray(trans).tolist(),
        }

        # and save it to a file
        with open("calibration_matrix.yaml", "w") as f:
            yaml.dump(data, f)

        #РЕТИФИКАЦИЯ СТЕРЕОПАРЫ

        rectifyScale= 1
        rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))

        stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
        stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

        print("Saving parameters!")
        cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)

        cv_file.write('stereoMapL_x',stereoMapL[0])
        cv_file.write('stereoMapL_y',stereoMapL[1])
        cv_file.write('stereoMapR_x',stereoMapR[0])
        cv_file.write('stereoMapR_y',stereoMapR[1])

        cv_file.release()

        

class StartPattern(QThread):
    def __init__(self, mawindow, parent=None):
        super().__init__()
        self.mawindow = mawindow

    def pattern_onClicked(mawindow):
        #p = pattern_ver2.resize_process(self) #for Windows
        pattern = cv.imread('chessboard.png') 
        #cv2.imshow('Pattern', pattern)    
        p_res = cv.resize(pattern, (640, 480))
        cv.imshow('pattern', pattern)
        # res_image =np.asarray(p_res)
        # img = Image.fromarray(p_res, 'RGB')
        # pyplot.imshow(img)
        # pyplot.show()
         


class StopThread(QThread):  
    def __init__(self, maiwindow, parent=None):
        super().__init__()
        self.maiwindow = maiwindow

    def stop(maiwindow):
        AppInterface.fl = False        
        print(AppInterface.fl)
        # time.sleep(3)
        # AppInterface.fl = True  
     


class StartThread(QThread):  
    def __init__(self, mainwindow, parent=None):
        super().__init__()
        self.mainwindow = mainwindow
        
    def run(mainwindow):


        num = 0
        AppInterface.fl = True
            
        while(True):
            
            if(AppInterface.fl):

                ret, frame = cap.read()
                #cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
                num+=1
                left, right = decode(frame)
                

                # im1 = cv.cvtColor(left, cv.COLOR_BGR2RGB)
                # im2 = cv.cvtColor(right, cv.COLOR_BGR2RGB)
                # height1, width1, channel1 = im1.shape
                # height2, width2, channel2 = im2.shape
                # step1 = channel1 * width1
                # step2 = channel2 * width2
                # qImg1 = QImage(im1.data, width1, height1, step1, QImage.Format_RGB888)
                # qImg2 = QImage(im2.data, width2, height2, step2, QImage.Format_RGB888)
                # mainwindow.changePixmap1.emit(qImg1)
                # mainwindow.changePixmap2.emit(qImg2)
                # @QtCore.pyqtSlot(QImage)
                # .ui.widget.setPixmap(QPixmap.fromImage(qImg1))                      
                # self.ui.widget_2.setPixmap(QPixmap.fromImage(qImg2))
                

                cv.imwrite('/home/mikhail/Work/final/images/imageLeft/imageL' + str(num) + '.png', left)
                cv.imwrite('/home/mikhail/Work/final/images/imageRight/imageR' + str(num) + '.png', right)
                #
                print('success')
                time.sleep(1)
            else:
                break

                

        cap.release()
        cv.destroyAllWindows()
 
 

class AppInterface(QtWidgets.QWidget):
    
    #i = 1
    fl = False
    # chess = None
    # list_of_all_corners_left = list()
    # list_of_all_corners_right = list()
    
    # row = ''



    def __init__(self):

        
        QApplication.processEvents()
        # Это здесь нужно для доступа к переменным, методам
        super().__init__()
        self.ui = Interface.Ui_Form()
        self.ui.setupUi(self) # Это нужно для инициализации нашего дизайна
                  
        # События нажатия на кнопки
        self.ui.pushButton_3.clicked.connect(self.stPattern)
        self.ui.pushButton.clicked.connect(self.launchStart)
        self.ui.pushButton_2.clicked.connect(self.launchStop) 
        self.ui.pushButton_4.clicked.connect(self.startCalib)
        #self.ui.pushButton_5.clicked.connect(self.launchStop)      
        #self.ui.Add_Button.clicked.connect(self.add) 
        #self.ui.Edit_Button.clicked.connect(self.edit) 
        self.Startthread_instance = StartThread(mainwindow=self)
        self.Stopthread_instance = StopThread(maiwindow=self)
        self.StartPattern_instance = StartPattern(mawindow=self)
        self.CalibrateCamera_instance = Calibration(calwindow=self)
        self.CalibrateCamera_instance.changePixmap1L.connect(self.setImage1)
        self.CalibrateCamera_instance.changePixmap2R.connect(self.setImage2)
        self.th1 = Thread1(self)
        self.th1.changePixmap1.connect(self.setImage1)
        self.th1.changePixmap2.connect(self.setImage2)
        self.th1.start()
        
        


     
        
    @QtCore.pyqtSlot(QImage)
    def setImage1(self, qImg1):
        self.ui.widget.setPixmap(QPixmap.fromImage(qImg1))

    @QtCore.pyqtSlot(QImage)
    def setImage2(self, qImg2):
        self.ui.widget_2.setPixmap(QPixmap.fromImage(qImg2))
  
    def launchStart(self):
        self.Startthread_instance.start()
        
       
    
    def launchStop(self):
        self.Stopthread_instance.stop()
      
    def stPattern(self):
        self.StartPattern_instance.pattern_onClicked()
        
    def get_all_corners(self):
        return (self.list_of_all_corners_left, self.list_of_all_corners_right)
                 
    def startCalib(self):
        self.CalibrateCamera_instance.calibration()



def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = AppInterface()  # Создаём объект класса App
    window.show()  # Показываем окно
    app.exec_()  # и запускаем приложение

if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()