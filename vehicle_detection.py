import numpy as np
import cv2
import os
import glob
import time
import logging
import queue


#显示中间过程图
def showDebugImage(images):
    import matplotlib.pyplot as plt
    imageTotal = len(images)
    for i in range(imageTotal):
        plt.subplot(1, imageTotal, i + 1)
        plt.title(images[i]["name"])
        plt.imshow(images[i]["img"], cmap="gray")
    plt.show()


#生成时间戳用作保存结果名
def getTimeString():
    times = time.localtime()
    timeStr = str(times.tm_year) + str(times.tm_mon).zfill(2) + str(times.tm_mday).zfill(2) + \
        str(times.tm_hour).zfill(2) + str(times.tm_min).zfill(2)
    return timeStr


class Marker(object):
    def __init__(self, templateMaskPath, templateEdgePath, markerRoi):
        self.templateMask = None
        self.templateEdge = None
        self.templateMaskPath = templateMaskPath
        self.templateEdgePath = templateEdgePath
        self.markerRoi = markerRoi
        self.mask = None
        self.edge = None
        self.lowerThresh = 0
        self.upperThresh = 0
        self.binaryThresh = 0

        self.loadMarkerImage()
        self.updateMarkerThreshold()

    #加载marker模板图
    def loadMarkerImage(self):
        if self.templateMaskPath is not None:
            self.templateMask = cv2.imread(self.templateMaskPath, cv2.IMREAD_GRAYSCALE)

        if self.templateEdgePath is not None:
            self.templateEdge = cv2.imread(self.templateEdgePath, cv2.IMREAD_GRAYSCALE)

    #更新判断marker是否被遮挡相关的阈值
    def updateMarkerThreshold(self):
        if self.templateEdge is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilateEdge = cv2.dilate(self.templateEdge, kernel)
            lowerLimit = cv2.countNonZero(self.templateEdge) * 0.1
            upperLimit = cv2.countNonZero(dilateEdge)
            totalPixel = self.templateEdge.shape[0] * self.templateEdge.shape[1]
            upperLimit = upperLimit + (totalPixel - upperLimit) * 0.5
            self.lowerThresh = lowerLimit
            self.upperThresh = upperLimit

        if self.templateMask is not None:
            self.binaryThresh = int(int(self.markerRoi[2]) * int(self.markerRoi[3]) * 0.15)

class VehicleDetection(object):
    def __init__(self):
        #marker模板
        #XiLong1
        self.frontMarkerRoi = (264, 418, 48, 22)               
        self.rearMarkerRoi = (344, 400, 53, 23)                
        #XiLong2
        # self.frontMarkerRoi = (199, 398, 63, 22)                
        # self.rearMarkerRoi = (147, 377, 38, 38)                 
        self.frontMarkerMaskPath = './frontMarkerMask.tif'
        self.frontMarkerEdgePath = './frontMarkerEdge.tif'
        self.rearMarkerMaskPath = './rearMarkerMask.tif'
        self.rearMarkerEdgePath = './rearMarkerEdge.tif'
        self.frontMarker = Marker(self.frontMarkerMaskPath, self.frontMarkerEdgePath, self.frontMarkerRoi)
        self.rearMarker = Marker(self.rearMarkerMaskPath, self.rearMarkerEdgePath, self.rearMarkerRoi)

        #状态记录
        self.frameIndex = 0
        self.enterFrameIndex = 0
        self.leaveFrameIndex = 0
        self.vehicleExist = False   

        #通用设置
        self.frameInterval = 1                                   #处理帧间隔
        self.distractionSavePath = "./distraction"
        self.enterAndLeaveSavePath = "./result"
        self.withGradient = True
        self.showDebugImage = False
        self.whetherRotate = False
        self.curTime = getTimeString()

        #用于更新背景
        self.updateTemplateEnable = False
        self.updateTemplateInterval = 30 * 60 * 30      #更新一次模板间隔的帧数
        self.frameCount = 0                             #间隔计数
        self.saveDistraction = False
        self.collectMarkers = False
        self.frontMarkerQueue = queue.Queue(maxsize = 5)
        self.rearMarkerQueue = queue.Queue(maxsize = 5)

    
    #基于marker的梯度信息来判断marker是否被遮挡
    def whetherShieldWithGradient(self, marker:Marker, cropped):
        if len(cropped.shape) == 3:
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        cannyImg = cv2.Canny(cropped, 30, 70)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cannyImg = cv2.dilate(cannyImg, kernel)
        diff = marker.templateEdge & ~cannyImg

        if self.showDebugImage is True:
            images = []
            singleImg = {"name": "src", "img": cropped}
            images.append(singleImg)
            singleImg = {"name": "markerEdge", "img": marker.templateEdge}
            images.append(singleImg)
            singleImg = {"name": "curEdge", "img": cannyImg}
            images.append(singleImg)
            singleImg = {"name": "diff", "img": diff}
            images.append(singleImg)
            showDebugImage(images)

        countDiff = cv2.countNonZero(diff)
        countCur = cv2.countNonZero(cannyImg)
        if countDiff < marker.lowerThresh and countCur < marker.upperThresh:
            return False
        else:
            return True


    #基于marker的二值化图来判断marker是否被遮挡
    def whetherShieldWithThresh(self, marker:Marker, cropped):
        if len(cropped.shape) == 3:
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        ret, binary = cv2.threshold(cropped, 0, 255, cv2.THRESH_OTSU)
        if ret is False:
            logging.error("threshold失败！")
            return (False, None)
        diff = cv2.absdiff(binary, marker.templateMask)

        if self.showDebugImage is True:
            images = []
            singleImg = {"name": "src", "img": cropped}
            images.append(singleImg)
            singleImg = {"name": "marker", "img": marker.templateMask}
            images.append(singleImg)
            singleImg = {"name": "cur", "img": binary}
            images.append(singleImg)
            singleImg = {"name": "diff", "img": diff}
            images.append(singleImg)
            showDebugImage(images)

        noneZero = cv2.countNonZero(diff)
        if noneZero > marker.binaryThresh:
            return True
        else:
            return False


    #更新marker模板
    def updateMarkerTemplate(self):
        if not self.frontMarkerQueue.full() or not self.rearMarkerQueue.full():
            logging.error("未能收集足够的marker背景，更新背景失败！")
            return False
        
        scale = 1 / self.frontMarkerQueue.qsize() 
        frontMarker = None
        while not self.frontMarkerQueue.empty():
            if frontMarker is None:
                frontMarker = self.frontMarkerQueue.get() * scale
            else:
                frontMarker += self.frontMarkerQueue.get() * scale
        binaryThresh = int(255 * scale * 2) + 1    #背景图组中同一位置出现两次以上边缘点则标记入背景 
        _, self.frontMarker.templateEdge = cv2.threshold(frontMarker, binaryThresh, 255, cv2.THRESH_BINARY)

        scale = 1 / self.rearMarkerQueue.qsize() 
        rearMarker = None
        while not self.rearMarkerQueue.empty():
            if rearMarker is None:
                rearMarker = self.rearMarkerQueue.get() * scale
            else:
                rearMarker += self.rearMarkerQueue.get() * scale
        binaryThresh = int(255 * scale * 2) + 1
        _, self.rearMarker.templateEdge = cv2.threshold(rearMarker, binaryThresh, 255, cv2.THRESH_BINARY)
        logging.debug("成功更新marker背景！")

        return True


    #收集用于叠加生成背景的marker图
    def collectMarkerBackground(self, imgSrc, markerPosition):
        if markerPosition == 'front':
            markerEdge = self.getMarkerEdge(imgSrc, self.frontMarkerRoi)
            if self.frontMarkerQueue.full():
                self.frontMarkerQueue.get()
            self.frontMarkerQueue.put(markerEdge)        
        elif markerPosition == 'rear':
            markerEdge = self.getMarkerEdge(imgSrc, self.rearMarkerRoi)
            if self.rearMarkerQueue.full():
                self.rearMarkerQueue.get()
            self.rearMarkerQueue.put(markerEdge)        


    #检测车辆是否位于检测区域内
    def checkVehicleExist(self, frame):
        if self.whetherRotate is True:
            frame = np.rot90(frame, 3)

        #检测前参照物遮挡状态
        self.frameIndex += 1
        self.frameCount += 1
        (x, y, w, h) = self.frontMarkerRoi
        frontCrop = frame[int(y) : int(y) + int(h), int(x) : int(x) + int(w)]
        if self.withGradient is True:
            whetherShieldFlag = self.whetherShieldWithGradient(self.frontMarker, frontCrop)
        else:
            whetherShieldFlag = self.whetherShieldWithThresh(self.frontMarker, frontCrop)

        #车辆进入检测区域
        if whetherShieldFlag is True and self.vehicleExist is False:       
            if abs(self.frameIndex - self.leaveFrameIndex) < 5:    
                return self.vehicleExist
            (x, y, w, h) = self.rearMarkerRoi
            rearCrop = frame[int(y) : int(y) + int(h), int(x) : int(x) + int(w)]
            if self.withGradient is True:
                reviewFlag = self.whetherShieldWithGradient(self.rearMarker, rearCrop)  
            else:
                reviewFlag = self.whetherShieldWithThresh(self.rearMarker, rearCrop)
            if reviewFlag is True:
                self.vehicleExist = True
                self.enterFrameIndex = self.frameIndex
                cv2.imwrite(os.path.join(self.enterAndLeaveSavePath, self.curTime + "_" + str(self.frameIndex) + "_enterFrame.jpg"), frame)
                self.saveDistraction = True
                self.collectMarkers = True
        elif whetherShieldFlag is False and self.vehicleExist is True:
            if abs(self.frameIndex - self.enterFrameIndex) < 5:    #离开帧和进入帧相邻则认为是错误匹配
                return self.vehicleExist
            (x, y, w, h) = self.rearMarkerRoi
            rearCrop = frame[int(y) : int(y) + int(h), int(x) : int(x) + int(w)]
            if self.withGradient is True:
                reviewFlag = self.whetherShieldWithGradient(self.rearMarker, rearCrop)  
            else:
                reviewFlag = self.whetherShieldWithThresh(self.rearMarker, rearCrop)   #判断后marker是否被遮挡，只有两个marker都能找到才确定车辆离开
            if reviewFlag is True:
                if self.saveDistraction is True:         #同一辆车中间部分只保存一张图片
                    logging.debug("检测到拖车中间部分！")
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
            if self.updateTemplateEnable is True:
                if self.vehicleExist is False and abs(self.frameIndex - self.leaveFrameIndex) > 2 and self.collectMarkers is True:
                    self.collectMarkerBackground(frontCrop, "front")
                    self.collectMarkerBackground(rearCrop, "rear")
                    self.collectMarkers = False

                if self.frameCount >= self.updateTemplateInterval:
                    self.updateMarkerTemplate()
                    self.frameCount = 0

        return self.vehicleExist


    #生成对应marker的模板图，首次使用时需要手动调用
    def generateMarkerTemplate(self, src):
        self.frontMarker.templateEdge = self.getMarkerEdge(src, self.frontMarkerRoi)
        if self.frontMarker.templateEdge is not None:
            cv2.imwrite(self.frontMarkerEdgePath, self.frontMarker.templateEdge)
        self.frontMarker.templateMask = self.getMarkerMask(src, self.frontMarkerRoi)
        if self.frontMarker.templateMask is not None:
            cv2.imwrite(self.frontMarkerMaskPath, self.frontMarker.templateMask)

        self.rearMarker.templateEdge = self.getMarkerEdge(src, self.rearMarkerRoi)
        if self.rearMarker.templateEdge is not None:
            cv2.imwrite(self.rearMarkerEdgePath, self.rearMarker.templateEdge)

        self.rearMarker.templateMask = self.getMarkerMask(src, self.rearMarkerRoi)
        if self.rearMarker.templateMask is not None:
            cv2.imwrite(self.rearMarkerMaskPath, self.rearMarker.templateMask)


    #获取marker的二值化mask图
    def getMarkerMask(self, img, roi):
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (x, y, w, h) = roi
        cropped = img[int(y) : int(y) + int(h), int(x) : int(x) + int(w)]
        _, mask = cv2.threshold(cropped, 0, 255, cv2.THRESH_OTSU)

        if self.showDebugImage is True:
            images = []
            singleImg = {"name": "img", "img": img}
            images.append(singleImg)
            singleImg = {"name": "cropped", "img": cropped}
            images.append(singleImg)
            singleImg = {"name": "mask", "img": mask}
            images.append(singleImg)
            showDebugImage(images)

        return mask


    #获取marker的边缘图
    def getMarkerEdge(self, img, roi):
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (x, y, w, h) = roi
        cropped = img[int(y) : int(y) + int(h), int(x) : int(x) + int(w)]
        edgeImg = cv2.Canny(cropped, 50, 100)

        if self.showDebugImage is True:
            images = []
            singleImg = {"name": "img", "img": img}
            images.append(singleImg)
            singleImg = {"name": "cropped", "img": cropped}
            images.append(singleImg)
            singleImg = {"name": "edge", "img": edgeImg}
            images.append(singleImg)
            showDebugImage(images)

        return edgeImg


#运行图片集测试
def testWithImages():
    detectObj = VehicleDetection()
    inputDir = "./inputImgs/jpg/300"
    for dir in glob.glob("{}/*.jpg".format(inputDir)):
        logging.debug(dir)
        frame = cv2.imread(dir, cv2.IMREAD_COLOR)
        detectObj.checkVehicleExist(frame)


#运行本地视频测试
def testWithVideo():
    detectObj = VehicleDetection()
    inputDir = "./inputVideos/1110.avi"
    cap = cv2.VideoCapture(inputDir)
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        detectObj.checkVehicleExist(frame)


#运行摄像头测试
def testWithCamera():
    detectObj = VehicleDetection()
    detectObj.whetherRotate = True
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)    
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        detectObj.checkVehicleExist(frame)


if __name__ == '__main__':
    #初始化logging
    from logging import handlers
    logging.basicConfig(level=logging.DEBUG)
    th = handlers.TimedRotatingFileHandler(filename='all.log', when='D', backupCount=3, encoding='utf-8')
    th.setFormatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    logging.getLogger().addHandler(th)
    
    # testWithImages()
    testWithVideo()
    # testWithCamera()
    logging.debug('Test finish!')
