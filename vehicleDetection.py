import numpy as np
import cv2
import os
import glob
import time
import logging
import matplotlib.pyplot as plt
from logging import handlers, makeLogRecord


def showDebugImage(images):
    imageTotal = len(images)
    for i in range(imageTotal):
        plt.subplot(1, imageTotal, i + 1)
        plt.title(images[i]["name"])
        plt.imshow(images[i]["img"], cmap="gray")
    plt.show()


def getTimeString():
    times = time.localtime()
    timeStr = str(times.tm_year) + str(times.tm_mon).zfill(2) + str(times.tm_mday).zfill(2) + \
        str(times.tm_hour).zfill(2) + str(times.tm_min).zfill(2)
    return timeStr


class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        th.setFormatter(format_str)#设置文件里写入的格式
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)

log = Logger('all.log',level='debug')

class Marker(object):
    def __init__(self):
        self.templateMask = None
        self.templateEdge = None
        self.mask = None
        self.edge = None
        self.lowerThresh = 0
        self.upperThresh = 0
        self.binaryThresh = 0


class VehicleDetection(object):
    def __init__(self):
        self.frontMarkerRoi = (264, 418, 48, 22)                 #(x, y, w, h)
        self.rearMarkerRoi = (344, 400, 53, 23)                  #用于复判车辆是否真实离开
        self.frontMarkers = []                                   #用于更新背景
        self.rearMarkers = []                                    #用于更新背景
        self.frameInterval = 1                                   #选取帧间隔
        self.frontMarkerMask = None 
        self.rearMarkerMask = None
        self.withGradient = True
        self.showDebug = False
        self.whetherRotate = False
        self.frontMarkerThresh = 0
        self.frontMarkerUpperLimit = 0
        self.rearMarkerThresh = 0
        self.rearMarkerUpperLimit = 0
        self.distractionSavePath = "./distraction"
        self.enterAndLeaveSavePath = "./result"
        self.frontMarkerEdgePath = './frontMarkerEdge.tif'
        self.rearMarkerEdgePath = './rearMarkerEdge.tif'
        self.frontMarkerMaskPath = './frontMarkerEdge.tif'
        self.rearMarkerMaskPath = './rearMarkerEdge.tif'
        self.frameIndex = 0
        self.enterFrameIndex = 0
        self.leaveFrameIndex = 0
        self.curTime = getTimeString()
        self.vehicleExist = False                           
        self.saveDistraction = False
        self.collectMarkers = False

        self.loadMarkerTemplate()


    def loadMarkerTemplate(self):
        if self.withGradient is True:
            self.frontMarkerMask = cv2.imread(self.frontMarkerEdgePath, cv2.IMREAD_GRAYSCALE)
            self.rearMarkerMask = cv2.imread(self.rearMarkerEdgePath, cv2.IMREAD_GRAYSCALE)
        else:
            self.frontMarkerMask = cv2.imread(self.frontMarkerMaskPath, cv2.IMREAD_GRAYSCALE)
            self.rearMarkerMask = cv2.imread(self.rearMarkerMaskPath, cv2.IMREAD_GRAYSCALE)

        if self.frontMarkerMask is None or self.rearMarkerMask is None:
            log.logger.error("load marker mask fail!")
            exit()
        
        if self.withGradient is True:
            self.frontMarkerThresh, self.frontUpperLimitThresh = self.getMarkerThreshold(self.frontMarkerMask)
            self.rearMarkerThresh, self.rearUpperLimitThresh = self.getMarkerThreshold(self.rearMarkerMask)
        else:
            self.frontMarkerThresh = int(int(self.frontMarkerRoi[2]) * int(self.frontMarkerRoi[3]) * 0.15)
            self.rearMarkerThresh = int(int(self.rearMarkerRoi[2]) * int(self.rearMarkerRoi[3]) * 0.15)


    def getMarkerThreshold(self, markerEdge):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilateEdge = cv2.dilate(markerEdge, kernel)
        lowerLimit = cv2.countNonZero(markerEdge) * 0.1
        upperLimit = cv2.countNonZero(dilateEdge)
        totalPixel = markerEdge.shape[0] * markerEdge.shape[1]
        upperLimit = upperLimit + (totalPixel - upperLimit) * 0.5

        return lowerLimit, upperLimit


    def whetherShieldWithGradient(self, markerEdge, cropped, lowerLimit, upperLimit):
        if len(cropped.shape) == 3:
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        cannyImg = cv2.Canny(cropped, 30, 70)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cannyImg = cv2.dilate(cannyImg, kernel)
        diff = markerEdge & ~cannyImg

        if self.showDebug is True:
            images = []
            singleImg = {"name": "src", "img": cropped}
            images.append(singleImg)
            singleImg = {"name": "markerEdge", "img": markerEdge}
            images.append(singleImg)
            singleImg = {"name": "curEdge", "img": cannyImg}
            images.append(singleImg)
            singleImg = {"name": "diff", "img": diff}
            images.append(singleImg)
            showDebugImage(images)

        countDiff = cv2.countNonZero(diff)
        countCur = cv2.countNonZero(cannyImg)
        if countDiff < lowerLimit and countCur < upperLimit:
            return False
        else:
            return True


    #判断参照物是否被遮挡
    def whetherShieldWithThresh(self, markerMask, cropped, bainaryThresh):
        if len(cropped.shape) == 3:
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        ret, binary = cv2.threshold(cropped, 0, 255, cv2.THRESH_OTSU)
        if ret is False:
            log.logger.error("threshold失败！")
            return (False, None)
        diff = cv2.absdiff(binary, markerMask)

        if self.showDebug is True:
            images = []
            singleImg = {"name": "src", "img": cropped}
            images.append(singleImg)
            singleImg = {"name": "marker", "img": markerMask}
            images.append(singleImg)
            singleImg = {"name": "cur", "img": binary}
            images.append(singleImg)
            singleImg = {"name": "diff", "img": diff}
            images.append(singleImg)
            showDebugImage(images)

        noneZero = cv2.countNonZero(diff)
        if noneZero > bainaryThresh:
            return True
        else:
            return False


    def updateBackground(self):
        if len(self.frontMarkers) < 5 or len(self.rearMarkers) < 5:
            log.logger.error("更新marker背景失败！")
            return False
        
        frontMarker = 0.2 * self.frontMarkers[0] + 0.2 * self.frontMarkers[1] + 0.2 * self.frontMarkers[2] \
            + 0.2 * self.frontMarkers[3] + 0.2 * self.frontMarkers[4]
        rearMarker = 0.2 * self.rearMarkers[0] + 0.2 * self.rearMarkers[1] + 0.2 * self.rearMarkers[2] \
            + 0.2 * self.rearMarkers[3] + 0.2 * self.rearMarkers[4]

        global frontMarkerMask, rearMarkerMask
        _, frontMarkerMask = cv2.threshold(frontMarker, 120, 255, cv2.THRESH_BINARY)
        _, rearMarkerMask = cv2.threshold(rearMarker, 120, 255, cv2.THRESH_BINARY)
        log.logger.debug("更新marker背景！")

        self.frontMarkers.clear()
        self.rearMarkers.clear()

        return True


    def collectMarkerBackground(self, imgSrc, markerPosition):
        if len(self.frontMarkers) >= 5 or len(self.rearMarkers) >= 5:
            log.logger.debug("markers已收集满！")
            return False

        if markerPosition == 'front':
            markerEdge = self.getMarkerEdgeWithImage(imgSrc, self.frontMarkerRoi)
            self.frontMarkers.append(markerEdge)        
        elif markerPosition == 'rear':
            markerEdge = self.getMarkerEdgeWithImage(imgSrc, self.rearMarkerRoi)
            self.rearMarkers.append(markerEdge)        

        return True


    #检测车辆是否位于检测区域内
    def checkVehicleExist(self, frame):
        if self.whetherRotate is True:
            frame = np.rot90(frame, 3)

        #检测前参照物遮挡状态
        self.frameIndex += 1
        frontCrop = frame[int(self.frontMarkerRoi[1]) : int(self.frontMarkerRoi[1]) + int(self.frontMarkerRoi[3]), int(self.frontMarkerRoi[0]) \
                : int(self.frontMarkerRoi[0]) + int(self.frontMarkerRoi[2])]
        if self.withGradient is True:
            whetherShieldFlag = self.whetherShieldWithGradient(self.frontMarkerMask, frontCrop, self.frontMarkerThresh, self.frontUpperLimitThresh)
        else:
            whetherShieldFlag = self.whetherShieldWithThresh(self.frontMarkerMask, frontCrop, self.frontMarkerThresh)


        #车辆进入检测区域
        if whetherShieldFlag is True and self.vehicleExist is False:       
            if abs(self.frameIndex - self.leaveFrameIndex) < 5:    
                return self.vehicleExist

            rearCrop = frame[int(self.rearMarkerRoi[1]) : int(self.rearMarkerRoi[1]) + int(self.rearMarkerRoi[3]), int(self.rearMarkerRoi[0]) \
                    : int(self.rearMarkerRoi[0]) + int(self.rearMarkerRoi[2])]
            if self.withGradient is True:
                reviewFlag = self.whetherShieldWithGradient(self.rearMarkerMask, rearCrop, self.rearMarkerThresh, self.rearUpperLimitThresh)  
            else:
                reviewFlag = self.whetherShieldWithThresh(self.rearMarkerMask, rearCrop, self.rearMarkerThresh)
            if reviewFlag is True:
                self.vehicleExist = True
                self.enterFrameIndex = self.frameIndex
                cv2.imwrite(os.path.join(self.enterAndLeaveSavePath, self.curTime + "_" + str(self.frameIndex) + "_enterFrame.jpg"), frame)
                self.saveDistraction = True
        elif whetherShieldFlag is False and self.vehicleExist is True:
            if abs(self.frameIndex - self.enterFrameIndex) < 5:    #离开帧和进入帧相邻则认为是错误匹配
                return self.vehicleExist
            
            rearCrop = frame[int(self.rearMarkerRoi[1]) : int(self.rearMarkerRoi[1]) + int(self.rearMarkerRoi[3]), int(self.rearMarkerRoi[0]) \
                : int(self.rearMarkerRoi[0]) + int(self.rearMarkerRoi[2])]
            if self.withGradient is True:
                reviewFlag = self.whetherShieldWithGradient(self.rearMarkerMask, rearCrop, self.rearMarkerThresh, self.rearUpperLimitThresh)  
            else:
                reviewFlag = self.whetherShieldWithThresh(self.rearMarkerMask, rearCrop, self.rearMarkerThresh)   #判断后marker是否被遮挡，只有两个marker都能找到才确定车辆离开

            if reviewFlag is True:
                if self.saveDistraction is True:         #同一辆车中间部分只保存一张图片
                    log.logger.debug("检测到拖车中间部分！")
                    cv2.imwrite(os.path.join(self.distractionSavePath, self.curTime + "_" + str(self.frameIndex) + "_distractionFrame.jpg"), frame)
                    self.saveDistraction = False
            else:
                self.vehicleExist = False
                self.leaveFrameIndex = self.frameIndex
                cv2.imwrite(os.path.join(self.enterAndLeaveSavePath, self.curTime + "_" + str(self.frameIndex) + "_leaveFrame.jpg"), frame)
            
            #更新图片保存序号
            if self.frameIndex > 5000:
                self.frameIndex = 0
                self.curTime = getTimeString()

            #更新背景
            if self.vehicleExist is False and abs(self.frameIndex - self.leaveFrameIndex) > 2 and self.collectMarkers is True:
                self.collectMarkerBackground(frontCrop, "front")
                self.collectMarkerBackground(rearCrop, "rear")
                self.collectMarkers = False

            if len(self.frontMarkers) >= 5 and len(self.rearMarkers) >= 5:
                self.updateBackground()

        return self.vehicleExist


    def writeMarkerImage(self, srcPath):
        src = cv2.imread(srcPath, cv2.IMREAD_GRAYSCALE)
        if self.withGradient is True:
            frontMarkerEdge = self.getMarkerEdgeWithImage(src, self.frontMarkerRoi)
            if frontMarkerEdge is None:
                print("get front marker fail!")
                exit()
            cv2.imwrite("./frontMarkerEdge.tif", frontMarkerEdge)

            rearMarkerEdge = self.getMarkerEdgeWithImage(src, self.rearMarkerRoi)
            if rearMarkerEdge is None:
                print("get rear marker fail!")
                exit()
            cv2.imwrite("./rearMarkerEdge.tif", rearMarkerEdge)
        else:
            frontMarkerMask = self.getMarkerMaskWithImage(src, self.frontMarkerRoi)
            if frontMarkerMask is None:
                print("get front marker fail!")
                exit()
            cv2.imwrite("./frontMarkerMask.tif", frontMarkerMask)

            rearMarkerMask = self.getMarkerMaskWithImage(src, self.rearMarkerRoi)
            if rearMarkerMask is None:
                print("get rear marker fail!")
                exit()
            cv2.imwrite("./rearMarkerMask.tif", rearMarkerMask)


    def getMarkerMaskWithImage(self, img, roi):
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (x, y, w, h) = roi
        cropped = img[int(y) : int(y) + int(h), int(x) : int(x) + int(w)]
        _, mask = cv2.threshold(cropped, 0, 255, cv2.THRESH_OTSU)

        if self.showDebug is True:
            plt.subplot(1, 3, 1)
            plt.imshow(img, cmap="gray")
            plt.title("img")
            plt.subplot(1, 3, 2)
            plt.imshow(cropped, cmap="gray")
            plt.title("cropped")
            plt.subplot(1, 3, 3)
            plt.imshow(mask, cmap="gray")
            plt.title("mask")
            plt.show()

        return mask


    def getMarkerEdgeWithImage(self, img, roi):
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (x, y, w, h) = roi
        cropped = img[int(y) : int(y) + int(h), int(x) : int(x) + int(w)]
        edgeImg = cv2.Canny(cropped, 50, 100)

        if self.showDebug is True:
            plt.subplot(1, 3, 1)
            plt.imshow(img, cmap="gray")
            plt.title("img")
            plt.subplot(1, 3, 2)
            plt.imshow(cropped, cmap="gray")
            plt.title("cropped")
            plt.subplot(1, 3, 3)
            plt.imshow(edgeImg, cmap="gray")
            plt.title("edge")
            plt.show()

        return edgeImg


def testWithImages():
    detectObj = VehicleDetection()
    inputDir = "./inputImgs/jpg/300"
    for dir in glob.glob("{}/*.jpg".format(inputDir)):
        log.logger.debug(dir)
        frame = cv2.imread(dir, cv2.IMREAD_COLOR)
        detectObj.checkVehicleExist(frame)


def testWithVideo():
    detectObj = VehicleDetection()
    inputDir = "./inputVideos/1110.avi"
    cap = cv2.VideoCapture(inputDir)
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        detectObj.checkVehicleExist(frame)


def testWithCamera():
    detectObj = VehicleDetection()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)    
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        detectObj.checkVehicleExist(frame)


if __name__ == '__main__':
    testWithImages()
    # testWithVideo()
    # testWithCamera()
    print('Test finish!')