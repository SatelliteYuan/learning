# from PIL.Image import ImageTransformHandler
from matplotlib.image import imread
import numpy as np
import cv2
import os
import glob
import time
import logging
import matplotlib.pyplot as plt
from logging import handlers, makeLogRecord

# XiLong1
# frontMarkerRoi = (264, 418, 48, 22)                 #(x, y, w, h)
# rearMarkerRoi = (344, 400, 53, 23)                 #用于复判车辆是否真实离开

# # XiLong2
frontMarkerRoi = (199, 398, 63, 22)                 #(x, y, w, h)
rearMarkerRoi = (147, 377, 38, 38)                  #用于复判车辆是否真实离开

frontMarkers = []                                   #用于更新背景
rearMarkers = []                                    #用于更新背景

frameInterval = 1                               #选取帧间隔
frontMarkerMask = None 
rearMarkerMask = None
distractionDir = "./distraction"
enterAndLeaveDir = "./result"
withGradient = True
showDebug = False


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

def showDebugImage(images):
    imageTotal = len(images)
    for i in range(imageTotal):
        plt.subplot(1, imageTotal, i + 1)
        plt.title(images[i]["name"])
        plt.imshow(images[i]["img"], cmap="gray")
    plt.show()


def getMarkerThreshold(markerEdge):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilateEdge = cv2.dilate(markerEdge, kernel)
    lowerLimit = cv2.countNonZero(markerEdge) * 0.1
    upperLimit = cv2.countNonZero(dilateEdge)
    totalPixel = markerEdge.shape[0] * markerEdge.shape[1]
    upperLimit = upperLimit + (totalPixel - upperLimit) * 0.5

    return lowerLimit, upperLimit


def whetherShieldWithGradient(markerEdge, cropped, thresh, upperLimitThresh):
    if len(cropped.shape) == 3:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    cannyImg = cv2.Canny(cropped, 30, 70)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cannyImg = cv2.dilate(cannyImg, kernel)
    diff = markerEdge & ~cannyImg

    if showDebug is True:
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
    if countDiff < thresh and countCur < upperLimitThresh:
        return False
    else:
        return True


#判断参照物是否被遮挡
def whetherShieldWithThresh(markerMask, cropped, thresh):
    if len(cropped.shape) == 3:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    ret, binary = cv2.threshold(cropped, 0, 255, cv2.THRESH_OTSU)
    if ret is False:
        log.logger.error("threshold失败！")
        return (False, None)
    diff = cv2.absdiff(binary, markerMask)

    if showDebug is True:
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
    if noneZero > thresh:
        return True
    else:
        return False


def getTimeString():
    times = time.localtime()
    timeStr = str(times.tm_year) + str(times.tm_mon).zfill(2) + str(times.tm_mday).zfill(2) + \
        str(times.tm_hour).zfill(2) + str(times.tm_min).zfill(2)
    return timeStr


def updateBackground():
    if len(frontMarkers) < 5 or len(rearMarkers) < 5:
        log.logger.error("更新marker背景失败！")
        return False
    
    frontMarker = 0.2 * frontMarkers[0] + 0.2 * frontMarkers[1] + 0.2 * frontMarkers[2] \
        + 0.2 * frontMarkers[3] + 0.2 * frontMarkers[4]
    rearMarker = 0.2 * rearMarkers[0] + 0.2 * rearMarkers[1] + 0.2 * rearMarkers[2] \
        + 0.2 * rearMarkers[3] + 0.2 * rearMarkers[4]

    global frontMarkerMask, rearMarkerMask
    _, frontMarkerMask = cv2.threshold(frontMarker, 120, 255, cv2.THRESH_BINARY)
    _, rearMarkerMask = cv2.threshold(rearMarker, 120, 255, cv2.THRESH_BINARY)
    log.logger.debug("更新marker背景！")

    frontMarkers.clear()
    rearMarkers.clear()

    return True


def collectMarkerBackground(imgSrc, markerPosition):
    global frontMarkers, rearMarkers
    if len(frontMarkers) >= 5 or len(rearMarkers) >= 5:
        log.logger.debug("markers已收集满！")
        return False

    if markerPosition == 'front':
        markerEdge = getMarkerEdgeWithImage(imgSrc, frontMarkerRoi)
        frontMarkers.append(markerEdge)        
    elif markerPosition == 'rear':
        markerEdge = getMarkerEdgeWithImage(imgSrc, rearMarkerRoi)
        rearMarkers.append(markerEdge)        

    return True


#检测车辆是否位于检测区域内
def checkVehicleExist(inputPath):
    frameIndex = 0
    enterFrameIndex = 0
    leaveFrameIndex = 0
    curTime = getTimeString()
    vehicleExist = False                           
    saveDistraction = False
    collectMarkers = False

    #设置成自动曝光
    if inputPath is None:
        video = cv2.VideoCapture(0)
        video.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)    
    else:
        video = cv2.VideoCapture(inputPath)

    if withGradient is True:
        frontMarkerThresh, frontUpperLimitThresh = getMarkerThreshold(frontMarkerMask)
        rearMarkerThresh, rearUpperLimitThresh = getMarkerThreshold(rearMarkerMask)
    else:
        frontMarkerThresh = int(int(frontMarkerRoi[2]) * int(frontMarkerRoi[3]) * 0.15)
        rearMarkerThresh = int(int(rearMarkerRoi[2]) * int(rearMarkerRoi[3]) * 0.15)

    while True:
        ret, frame = video.read()
        if ret is False:
            break
        frame = np.rot90(frame, 3)
    # for dir in glob.glob("./inputImgs/jpg/300/*.jpg"):
    #     frame = cv2.imread(dir, cv2.IMREAD_COLOR)
        #检测前参照物遮挡状态
        frameIndex += 1
        frontCrop = frame[int(frontMarkerRoi[1]) : int(frontMarkerRoi[1]) + int(frontMarkerRoi[3]), int(frontMarkerRoi[0]) \
                : int(frontMarkerRoi[0]) + int(frontMarkerRoi[2])]
        if withGradient is True:
            whetherShieldFlag = whetherShieldWithGradient(frontMarkerMask, frontCrop, frontMarkerThresh, frontUpperLimitThresh)
        else:
            whetherShieldFlag = whetherShieldWithThresh(frontMarkerMask, frontCrop, frontMarkerThresh)


        #车辆进入检测区域
        if whetherShieldFlag is True and vehicleExist is False:       
            if abs(frameIndex - leaveFrameIndex) < 5:    
                continue
            rearCrop = frame[int(rearMarkerRoi[1]) : int(rearMarkerRoi[1]) + int(rearMarkerRoi[3]), int(rearMarkerRoi[0]) \
                : int(rearMarkerRoi[0]) + int(rearMarkerRoi[2])]
            if withGradient is True:
                reviewFlag = whetherShieldWithGradient(rearMarkerMask, rearCrop, rearMarkerThresh, rearUpperLimitThresh)  
            else:
                reviewFlag = whetherShieldWithThresh(rearMarkerMask, rearCrop, rearMarkerThresh)
            if reviewFlag is True:
                vehicleExist = True
                enterFrameIndex = frameIndex
                cv2.imwrite(os.path.join(enterAndLeaveDir, curTime + "_" + str(frameIndex) + "_enterFrame.jpg"), frame)
                saveDistraction = True
        elif whetherShieldFlag is False and vehicleExist is True:
            if abs(frameIndex - enterFrameIndex) < 5:    #离开帧和进入帧相邻则认为是错误匹配
                continue
            rearCrop = frame[int(rearMarkerRoi[1]) : int(rearMarkerRoi[1]) + int(rearMarkerRoi[3]), int(rearMarkerRoi[0]) \
                : int(rearMarkerRoi[0]) + int(rearMarkerRoi[2])]
            if withGradient is True:
                reviewFlag = whetherShieldWithGradient(rearMarkerMask, rearCrop, rearMarkerThresh, rearUpperLimitThresh)  
            else:
                reviewFlag = whetherShieldWithThresh(rearMarkerMask, rearCrop, rearMarkerThresh)   #判断后marker是否被遮挡，只有两个marker都能找到才确定车辆离开

            if reviewFlag is True:
                if saveDistraction is True:         #同一辆车中间部分只保存一张图片
                    log.logger.debug("检测到拖车中间部分！")
                    cv2.imwrite(os.path.join(distractionDir, curTime + "_" + str(frameIndex) + "_distractionFrame.jpg"), frame)
                    saveDistraction = False
                continue
            
            vehicleExist = False
            leaveFrameIndex = frameIndex
            cv2.imwrite(os.path.join(enterAndLeaveDir, curTime + "_" + str(frameIndex) + "_leaveFrame.jpg"), frame)
        
        #更新图片保存序号
        if frameIndex > 5000:
            frameIndex = 0
            curTime = getTimeString()

        #更新背景
        if vehicleExist is False and abs(frameIndex - leaveFrameIndex) > 2 and collectMarkers is True:
            collectMarkerBackground(frontCrop, "front")
            collectMarkerBackground(rearCrop, "rear")
            collectMarkers = False

        if len(frontMarkers) >= 5 and len(rearMarkers) >= 5:
            updateBackground()

def writeMarkerImage():
    src = cv2.imread("./frames0.tif", cv2.IMREAD_GRAYSCALE)
    if withGradient is True:
        frontMarkerEdge = getMarkerEdgeWithImage(src, frontMarkerRoi)
        if frontMarkerEdge is None:
            print("get front marker fail!")
            exit()
        cv2.imwrite("./frontMarkerEdge.tif", frontMarkerEdge)

        rearMarkerEdge = getMarkerEdgeWithImage(src, rearMarkerRoi)
        if rearMarkerEdge is None:
            print("get rear marker fail!")
            exit()
        cv2.imwrite("./rearMarkerEdge.tif", rearMarkerEdge)
    else:
        frontMarkerMask = getMarkerMaskWithImage(src, frontMarkerRoi)
        if frontMarkerMask is None:
            print("get front marker fail!")
            exit()
        cv2.imwrite("./frontMarkerMask.tif", frontMarkerMask)

        rearMarkerMask = getMarkerMaskWithImage(src, rearMarkerRoi)
        if rearMarkerMask is None:
            print("get rear marker fail!")
            exit()
        cv2.imwrite("./rearMarkerMask.tif", rearMarkerMask)


def getMarkerMaskWithImage(img, roi):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (x, y, w, h) = roi
    cropped = img[int(y) : int(y) + int(h), int(x) : int(x) + int(w)]
    _, mask = cv2.threshold(cropped, 0, 255, cv2.THRESH_OTSU)

    if showDebug is True:
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


def getMarkerEdgeWithImage(img, roi):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (x, y, w, h) = roi
    cropped = img[int(y) : int(y) + int(h), int(x) : int(x) + int(w)]
    edgeImg = cv2.Canny(cropped, 50, 100)

    if showDebug is True:
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


def testImages():
    inputDir = "./inputImgs/jpg/1123"

    if withGradient is True:
        frontMarkerThresh, frontUpperLimitThresh = getMarkerThreshold(frontMarkerMask)
        rearMarkerThresh, rearUpperLimitThresh = getMarkerThreshold(rearMarkerMask)
    else:
        frontMarkerThresh = int(int(frontMarkerRoi[2]) * int(frontMarkerRoi[3]) * 0.15)
        rearMarkerThresh = int(int(rearMarkerRoi[2]) * int(rearMarkerRoi[3]) * 0.15)

    for dir in glob.glob("{}/*.jpg".format(inputDir)):
        log.logger.debug(dir)
        frame = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)
        cropping = frame[int(frontMarkerRoi[1]) : int(frontMarkerRoi[1]) + int(frontMarkerRoi[3]), int(frontMarkerRoi[0]) \
                : int(frontMarkerRoi[0]) + int(frontMarkerRoi[2])]
        if withGradient is True:
            whetherShieldFlag = whetherShieldWithGradient(frontMarkerMask, cropping, frontMarkerThresh, frontUpperLimitThresh)
        else:
            whetherShieldFlag = whetherShieldWithThresh(frontMarkerMask, cropping, frontMarkerThresh)

        if whetherShieldFlag is False:
            log.logger.debug("frontMarker未被遮挡！")
            cropping = frame[int(rearMarkerRoi[1]) : int(rearMarkerRoi[1]) + int(rearMarkerRoi[3]), int(rearMarkerRoi[0]) \
                : int(rearMarkerRoi[0]) + int(rearMarkerRoi[2])]
            if withGradient is True:
                reviewFlag = whetherShieldWithGradient(rearMarkerMask, cropping, rearMarkerThresh, rearUpperLimitThresh)  
            else:
                reviewFlag = whetherShieldWithThresh(rearMarkerMask, cropping, rearMarkerThresh) 
            if reviewFlag is False:
                log.logger.debug("rearMakrer未被遮挡， 车辆已离开！")
            else:
                log.logger.debug("rearMaker被遮挡，检测道拖车中间部分")
        else:
            log.logger.debug("frontMarker被遮挡！")
            cropping = frame[int(rearMarkerRoi[1]) : int(rearMarkerRoi[1]) + int(rearMarkerRoi[3]), int(rearMarkerRoi[0]) \
                : int(rearMarkerRoi[0]) + int(rearMarkerRoi[2])]
            if withGradient is True:
                reviewFlag = whetherShieldWithGradient(rearMarkerMask, cropping, rearMarkerThresh, rearUpperLimitThresh)  
            else:
                reviewFlag = whetherShieldWithThresh(rearMarkerMask, cropping, rearMarkerThresh) 
            if reviewFlag is False:
                log.logger.debug("rearMakrer未被遮挡， 车辆未进场！")
            else:
                log.logger.debug("rearMaker被遮挡，车辆进场")
    

def main():
    if os.path.isdir(enterAndLeaveDir) is False:
        os.mkdir(enterAndLeaveDir)

    if os.path.isdir(distractionDir) is False:
        os.mkdir(distractionDir)

    # saveMakerImage()

    global frontMarkerMask, rearMarkerMask
    if withGradient is True:
        frontMarkerMask = cv2.imread("./frontMarkerEdge.tif", cv2.IMREAD_GRAYSCALE)
        rearMarkerMask = cv2.imread("./rearMarkerEdge.tif", cv2.IMREAD_GRAYSCALE)
    else:
        frontMarkerMask = cv2.imread("./frontMarkerMask.tif", cv2.IMREAD_GRAYSCALE)
        rearMarkerMask = cv2.imread("./rearMarkerMask.tif", cv2.IMREAD_GRAYSCALE)

    if frontMarkerMask is None or rearMarkerMask is None:
        log.logger.error("load marker mask fail!")
        exit()
   
    inputDir = None
    # inputDir = './inputVideos/1110.avi'
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

