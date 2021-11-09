from types import coroutine
import cv2
import os
import glob
from tqdm import tqdm
import time


# frontReferenceObject = None                     #前参照物二值化图
# rearReferenceObject = None                      #后参照物二值化图
markerMask = None
# frontReferenceObjectRoi = (766, 796, 193, 242)  #前参照物位置信息 (top, bottom, left, right)
# rearReferenceObjectRoi = (736, 778, 545, 581)   #后参照物位置
markerRoi = (250, 404, 64, 26)                  #(x, y, w, h)
frameInterval = 1                              #选取帧间隔
areaThresh = 200                                 #用于判断是否遮挡的面积阈值


#判断参照物是否被遮挡
def whetherShield(referenceObject, cropped):
    if len(cropped.shape) == 3:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    # diff = cv2.absdiff(referenceObject, cropped)
    # ret, diffBin = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    # contours, hierarchy = cv2.findContours(diffBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # if len(contours) > 0:
    #     for contour in contours:
    #         contourArea = cv2.contourArea(contour)
    #         if contourArea > 20:
    #             whetherShileldFlag = False
    #             break
    ret, binary = cv2.threshold(cropped, 0, 255, cv2.THRESH_OTSU)
    # noneZero = cv2.countNonZero(cropped)
    # if noneZero > 600 or noneZero < 300:
    #     return False

    diff = cv2.absdiff(binary, referenceObject)
    # cv2.namedWindow()
    # cv2.imshow("cropped", cropped)
    # cv2.imshow("referenceObject", referenceObject)
    # cv2.imshow("diff", diff)
    # cv2.waitKey(0)

    noneZero = cv2.countNonZero(diff)
    if noneZero < areaThresh:
        return (False, binary)

    return (True, binary)


#检测车辆是否位于检测区域内
def checkVehicleExist(inputPath, outputPath):
    video = cv2.VideoCapture(inputPath)
    vehIndex = os.path.splitext(os.path.basename(inputPath))[0]
    # frontReferenceObjectShield = False              #标识前参照物是否被遮挡
    # rearReferenceObjectShield = False               #标识后参照物是否被遮挡
    frameIndex = 0
    enterFrameIndex = 0
    startTime = time.time()
    (x, y, w, h) = markerRoi
    global areaThresh 
    vehicleExist = False                            #标识车辆是否在检测区域内   
    areaThresh = w * h * 0.2

    while True:
        ret, frame = video.read()
        if ret is False:
            break
        # os.makedirs('./imagesplicer_debug/{}'.format(vehIndex), exist_ok=True)
        # cv2.imwrite(os.path.join("imagesplicer_debug", vehIndex, str(frameIndex) + ".jpg"), frame)

        #检测前参照物遮挡状态
        cropping = frame[int(y) : int(y) + int(h), int(x) : int(x) + int(w)]
        whetherShieldFlag, binary = whetherShield(markerMask, cropping)
        if whetherShieldFlag is True and vehicleExist is False:       #车辆进入检测区域
            vehicleExist = True
            enterFrameIndex = frameIndex
            if frameIndex == 0:
                continue
            # cv2.imshow("crop", cropping)
            # cv2.imshow("binary", binary)
            # cv2.imshow("frame", frame)
            # cv2.waitKey(0)
            cv2.imwrite(os.path.join(outputPath, vehIndex + "_" + str(frameIndex) + "_enterCrop.tif"), cropping)
            cv2.imwrite(os.path.join(outputPath, vehIndex + "_" + str(frameIndex) + "_enterBin.tif"), binary)
            cv2.imwrite(os.path.join(outputPath, vehIndex + "_" + str(frameIndex) + "_enterFrame.jpg"), frame)
        elif whetherShieldFlag is False and vehicleExist is True:
            if frameIndex - enterFrameIndex < 10:    #离开帧和进入帧相邻则认为是错误匹配
                 continue
            vehicleExist = False
            # cv2.imshow("crop", cropping)
            # cv2.imshow("binary", binary)
            # cv2.imshow("frame", frame)
            # cv2.waitKey(0)
            cv2.imwrite(os.path.join(outputPath, vehIndex + "_" + str(frameIndex) + "_leaveBin.tif"), binary)
            cv2.imwrite(os.path.join(outputPath, vehIndex + "_" + str(frameIndex) + "_leaveCrop.tif"), cropping)
            cv2.imwrite(os.path.join(outputPath, vehIndex + "_" + str(frameIndex) + "_leaveFrame.jpg"), frame)
        #检测后参照物遮挡状态
        # top, bottom, left, right = rearReferenceObjectRoi
        # cropped = frame[int(top) : int(bottom), int(left) : int(right)]
        # whetherShieldFlag, binary = whetherShield(rearReferenceObject, cropped)
        # if whetherShieldFlag is False and rearReferenceObjectShield is True:        #车辆离开检测区域
        #     vehicleExist = False
        #     rearReferenceObjectShield = False
        #     # cv2.imwrite(os.path.join(outputPath, vehIndex + "_" + str(frameIndex) + "_leaveFrame.jpg"), frame)
        # elif whetherShieldFlag is True and rearReferenceObjectShield is False:
        #     rearReferenceObjectShield = True

        frameIndex += 1

def getMarkerTemplate():
    inputDir = './test_video_dir/1/fragment/91.avi'
    index = 0
    cap = cv2.VideoCapture(inputDir)
    templteFrameIndex = 53
    saveOriginalImage = False
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        if saveOriginalImage is True:
            cv2.imwrite("./test_video_dir/jihua/originalImage/" + str(index) + "_original.tif", frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (x, y, w, h) = markerRoi
        cropping = frame[int(y) : int(y) + int(h), int(x) : int(x) + int(w)]
        ret, binary = cv2.threshold(cropping, 0, 255, cv2.THRESH_OTSU)
        if index == templteFrameIndex:
            cv2.imwrite("inputImgs/markerMask.tif", binary)
            cv2.namedWindow("frame", cv2.WINDOW_FREERATIO)
            cv2.imshow("frame", frame)
            cv2.imshow("cropping", cropping)
            cv2.imshow("bin", binary)
            cv2.waitKey()
            break
        index += 1


if __name__ == '__main__':
    outputPath = './DebugResult'
    if os.path.isdir(outputPath) is False:
        os.mkdir(outputPath)

    # getMarkerTemplate()
    

    # imgPath = 'imagesplicer_debug'
    # frameId = 0
    # for path in glob.glob('{}/*.jpg'.format(imgPath)):
    #     cropped = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #     diff = cv2.absdiff(referenceObject, cropped)
    #     ret, diffBin = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    #     contours, hierarchy = cv2.findContours(diffBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     file_path = os.path.join(imgPath, str(frameId) + ".jpg")
    #     cv2.imwrite(file_path, diff)
    #     cv2.imshow("diffBin", diffBin)
    #     cv2.imshow("diff", diff)
    #     cv2.waitKey(0)
    #     frameId += 1


    # rearReferenceObject = cv2.imread(os.path.join("inputImgs", "rearReferenceObject.tif"), cv2.IMREAD_GRAYSCALE)
    markerMask = cv2.imread("./inputImgs/markerMask.tif", cv2.IMREAD_GRAYSCALE)
    if markerMask is None:
        print("load marker image fail!")
        exit()

    # cv2.imshow("rearReferenceObject", rearReferenceObject)
    # cv2.imshow("frontReferenceObject", frontReferenceObject)
    # cv2.waitKey(0)

    inputDir = "./test_video_dir/1"
    for inputPath in tqdm(glob.glob("{}/*.avi".format(inputDir))):
        checkVehicleExist(inputPath, outputPath)