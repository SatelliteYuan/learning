import numpy as np
import cv2
import os
import glob
import time
import logging
from logging import handlers


markerRoi = (250, 404, 64, 26)                  #(x, y, w, h)
marker2Roi = (147, 377, 38, 38)                 #用于复判车辆是否真实离开
frameInterval = 1                               #选取帧间隔
markerMask = cv2.imread("./markerMask.tif", cv2.IMREAD_GRAYSCALE)
markerMaskForReview = cv2.imread("./markerMaskForReview.tif", cv2.IMREAD_GRAYSCALE)
distractionDir = "./distraction"
enterAndLeaveDir = "./result"


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


#判断参照物是否被遮挡
def whetherShield(referenceObject, cropped, thresh):
    if len(cropped.shape) == 3:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    ret, binary = cv2.threshold(cropped, 0, 255, cv2.THRESH_OTSU)
    if ret is False:
        log.logger.error("threshold失败！")
        return (False, None)
    diff = cv2.absdiff(binary, referenceObject)

    # cv2.imshow("frame", referenceObject)
    # cv2.imshow("crop", cropped)
    # cv2.imshow("binary", binary)
    # cv2.imshow("diff", diff)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    noneZero = cv2.countNonZero(diff)
    if noneZero < thresh:
        return (False, binary)


    return (True, binary)


def getTimeString():
    times = time.localtime()
    timeStr = str(times.tm_year) + str(times.tm_mon).zfill(2) + str(times.tm_mday).zfill(2) + \
        str(times.tm_hour).zfill(2) + str(times.tm_min).zfill(2)
    return timeStr


#检测车辆是否位于检测区域内
def checkVehicleExist(inputPath):
    if inputPath is None:
        video = cv2.VideoCapture(0)
        video.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)    #设置成自动曝光
    else:
        video = cv2.VideoCapture(inputPath)
    frameIndex = 0
    enterFrameIndex = 0
    leaveFrameIndex = 0
    (x, y, w, h) = markerRoi
    curTime = getTimeString()
    vehicleExist = False                            #标识车辆是否在检测区域内   
    areaThresh = int(w * h * 0.15)
    distractionFlag = False

    while True:
        ret, frame = video.read()
        if ret is False:
            break
        frame = np.rot90(frame, 3)

        #检测前参照物遮挡状态
        frameIndex += 1
        cropping = frame[int(y) : int(y) + int(h), int(x) : int(x) + int(w)]
        whetherShieldFlag, binary = whetherShield(markerMask, cropping, areaThresh)
        # cv2.imshow("frame", frame)
        # cv2.imshow("crop", cropping)
        # cv2.waitKey()
        if whetherShieldFlag is True and vehicleExist is False:       #车辆进入检测区域
            if abs(frameIndex - leaveFrameIndex) < 5:    
                continue
            vehicleExist = True
            enterFrameIndex = frameIndex
            cv2.imwrite(os.path.join(enterAndLeaveDir, curTime + "_" + str(frameIndex) + "_enterFrame.jpg"), frame)
        elif whetherShieldFlag is False and vehicleExist is True:
            if abs(frameIndex - enterFrameIndex) < 5:    #离开帧和进入帧相邻则认为是错误匹配
                continue
            cropping = frame[int(marker2Roi[1]) : int(marker2Roi[1]) + int(marker2Roi[3]), int(marker2Roi[0]) \
                : int(marker2Roi[0]) + int(marker2Roi[2])]
            thresh = int(int(marker2Roi[2]) * int(marker2Roi[3]) * 0.15)
            reviewFlag, _ = whetherShield(markerMaskForReview, cropping, thresh)   #判断后marker是否被遮挡，只有两个marker都能找到才确定车辆离开
            if reviewFlag is True and distractionFlag is False:
                log.logger.debug("检测到拖车中间部分！")
                cv2.imwrite(os.path.join(distractionDir, curTime + "_" + str(frameIndex) + "_distractionFrame.jpg"), frame)
                distractionFlag = True
                continue
            
            distractionFlag = False
            vehicleExist = False
            leaveFrameIndex = frameIndex
            cv2.imwrite(os.path.join(enterAndLeaveDir, curTime + "_" + str(frameIndex) + "_leaveFrame.jpg"), frame)
        
        if frameIndex > 5000:
            frameIndex = 0
            curTime = getTimeString()

def getMarkerTemplate():
    index = 0
    cap = cv2.VideoCapture(0)
    templteFrameIndex = 53
    saveOriginalImage = False
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        if saveOriginalImage is True:
            cv2.imwrite("./test_video_dir/jihua/originalImage/" + str(index) + "_original.tif", frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = np.rot90(frame, 3)
        (x, y, w, h) = markerRoi
        cropping = frame[int(y) : int(y) + int(h), int(x) : int(x) + int(w)]
        ret, binary = cv2.threshold(cropping, 0, 255, cv2.THRESH_OTSU)
        if index == templteFrameIndex:
            cv2.imwrite("./src.tif", frame)
            cv2.imwrite("./markerMask.tif", binary)
            break
        index += 1


def getMarkerTemplateWithImage(img, roi):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (x, y, w, h) = roi
    cropped = img[int(y) : int(y) + int(h), int(x) : int(x) + int(w)]
    _, binary = cv2.threshold(cropped, 0, 255, cv2.THRESH_OTSU)
    cv2.imshow("cropped", cropped)
    cv2.imshow("img", img)
    cv2.imshow("binary", binary)
    cv2.waitKey(0)

    return binary


def testImages():
    inputDir = "./inputImgs"
    (x, y, w, h) = markerRoi
    # global areaThresh                            #标识车辆是否在检测区域内   
    areaThresh = w * h * 0.15

    for dir in glob.glob("{}/*.jpg".format(inputDir)):
        frame = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)
        cropping = frame[int(y) : int(y) + int(h), int(x) : int(x) + int(w)]
        whetherShieldFlag, binary = whetherShield(markerMask, cropping, areaThresh)
        if whetherShieldFlag is False:
            log.logger.debug("marker1未被遮挡！")
            cropping = frame[int(marker2Roi[1]) : int(marker2Roi[1]) + int(marker2Roi[3]), int(marker2Roi[0]) \
                : int(marker2Roi[0]) + int(marker2Roi[2])]
            thresh = int(marker2Roi[2]) * int(marker2Roi[3]) * 0.15
            reviewFlag, _ = whetherShield(markerMaskForReview, cropping, thresh)   #判断后marker是否被遮挡，只有两个marker都能找到才确定车辆离开
            if reviewFlag is False:
                log.logger.debug("marker2未被遮挡， 车辆已离开！")
            else:
                log.logger.debug("marker2被遮挡，检测道拖车中间部分")
                cv2.imshow("frame", frame)
                cv2.imshow("crop", cropping)
                cv2.imshow("binary", binary)
                cv2.waitKey(0)
    




def main():
    if os.path.isdir(enterAndLeaveDir) is False:
        os.mkdir(enterAndLeaveDir)

    if os.path.isdir(distractionDir) is False:
        os.mkdir(distractionDir)

    # getMarkerTemplate()

    # src = cv2.imread("./src.tif", cv2.IMREAD_GRAYSCALE)
    # frame = np.rot90(src, 3)
    # (x, y, w, h) = marker2Roi
    # cropping = frame[int(y) : int(y) + int(h), int(x) : int(x) + int(w)]
    # ret, binary = cv2.threshold(cropping, 0, 255, cv2.THRESH_OTSU)
    # gradient64 = cv2.Sobel(cropping,cv2.CV_64F,1,0)
    # gradient = cv2.convertScaleAbs(gradient64)
    # edge1 = cv2.Canny(cropping, 80, 150)
    # edge2 = cv2.Canny(cropping, 50, 100)
    # cv2.imwrite("./markerMask.tif", binary)
    # cv2.imshow("1", gradient)
    # # cv2.imshow("2", edge2)
    # cv2.imshow("crop", cropping)
    # cv2.waitKey(0)

    # src = cv2.imread("./src.tif", cv2.IMREAD_GRAYSCALE)
    # src = np.rot90(src, 3)
    # markerMask = getMarkerTemplateWithImage(src, markerRoi)
    # if markerMask is None:
    #     print("get mask fail!")
    #     exit()
    # cv2.imwrite("./markerMask.tif", markerMask)

    # markerMaskForReview = getMarkerTemplateWithImage(src, marker2Roi)
    # if markerMaskForReview is None:
    #     print("get mask fail!")
    #     exit()
    # cv2.imwrite("./markerMaskForReview.tif", markerMaskForReview)
    
    if markerMask is None or markerMaskForReview is None:
        log.logger.error("load marker mask fail!")
        exit()

    # inputDir = None
    inputDir = './inputVideos/1110.avi'
    checkVehicleExist(inputDir)

    # testImages()


log = Logger('all.log',level='debug')

if __name__ == '__main__':
    main()
    # try:
    #     main()
    # except Exception as e:
    #     log.logger.error('except', e)
    # finally:
    #     log.logger.debug('conclude')

