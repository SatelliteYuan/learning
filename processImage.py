import cv2
import numpy as np
import glob
import os
import time
import matplotlib.pyplot as plt
import copy

def calcGrayHist(img):
    # 计算灰度直方图
    h, w = img.shape[:2]
    grayHist = np.zeros([256], np.uint64)
    for i in range(h):
        for j in range(w):
            grayHist[img[i][j]] += 1
    return grayHist


def showGrayHist(grayHist):
    x = np.arange(256)
    # 绘制灰度直方图
    plt.plot(x, grayHist, 'r', linewidth=2, c='black')
    plt.xlabel("gray Label")
    plt.ylabel("number of pixels")
    plt.show()


def linearTransfer(img):
    h, w = img.shape[:2]
    out = np.zeros(img.shape, np.uint8)
    for i in range(h):
        for j in range(w):
            pix = img[i][j]
            if pix < 50:
                out[i][j] = 0.5 * pix
            elif pix < 150:
                out[i][j] = 3.6 * pix - 310
            else:
                out[i][j] = 0.238 * pix + 194
    # 数据类型转换
    out = np.around(out)
    out = out.astype(np.uint8)
    return out

def gammaTrasfer(img, gamma):
    # 图像归一化
    floatImg = img / 255.0

    # 伽马变换
    out = np.power(floatImg, gamma)
    out = np.uint8(out * 255)

    return out


def logTransfer(img, logParam):
    out = logParam * np.log(1.0 + img)
    out = np.uint8(out + 0.5)
    return out


# 分段线性变换
def SLT(img, x1, x2, y1, y2):
    lut = np.zeros(256)
    for i in range(256):
            if i < x1:
                lut[i] = (y1/x1)*i
            elif i < x2:
                lut[i] = ((y2-y1)/(x2-x1))*(i-x1)+y1
            else:
                lut[i] = ((y2-255.0)/(x2-255.0))*(i-255.0)+255.0
    img_output = cv2.LUT(img, lut)
    img_output = np.uint8(img_output+0.5)
    return img_output


def equalizeHist(img):
    if len(img.shape) > 2:
        out = copy.deepcopy(img)
        for j in range(3):
	        out[:, :, j] = cv2.equalizeHist(img[:,:,j])
    else:
        out = cv2.equalizeHist(img)

    return out

def clahe(img):  
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(img.shape) > 2:
        planes = cv2.split(img)
        merges = []
        for i in range(3):
            merges.append(clahe.apply(planes[i])) 
        dst = cv2.merge(merges)
    else:
        dst = clahe.apply(img)

    return dst

def splice():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    f = cap.get(cv2.CAP_PROP_FPS)
    b = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    print("f =", f, ",W =", w, ",h =", h, ",b=", b)
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        # w, h = frame.shape[:2]
        # print(w, h)
        cv2.imshow("frame", frame)
        cv2.waitKey(20)

    inputDir = "imagesplicer_debug/20210508160404/261.jpg"
    img = cv2.imread(inputDir, cv2.IMREAD_ANYCOLOR)
    left = img[:, 0 : 387]
    right = img[:, 388 : 720]
    # grayImg = cv2.imread(inputDir, cv2.IMREAD_GRAYSCALE)
    srcPoints2 = np.float32([[0, 194], [0, 818], [333, 779], [333, 274]])
    dstPoints2= np.float32([[387, 225], [387, 795], [720, 795], [720, 225]])
    # srcPoints = np.float32([[0, 251], [387, 194], [387, 818], [720, 779], [0 , 800], [720, 274]])
    # dstPoints= np.float32([[0,225], [387, 225], [387, 795], [720, 795], [0, 795], [720, 225]])
    srcPoints1 = np.float32([[0, 251], [387, 194], [387, 818], [0 , 800]])
    dstPoints1 = np.float32([[0, 225], [387, 225], [387, 795], [0, 795]])   
    # M = cv2.getPerspectiveTransform(srcPoints, dstPoints, )
    # M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC)
    # dst = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    M1, _ = cv2.findHomography(srcPoints1, dstPoints1, cv2.RANSAC) 
    dst1 = cv2.warpPerspective(left, M1, (left.shape[1], left.shape[0]))
    M2, _ = cv2.findHomography(srcPoints2, dstPoints2, cv2.RANSAC)
    dst = cv2.warpPerspective(right, M2, (img.shape[1], img.shape[0]))
    dst[:, 0 : 387] = dst1
    cv2.imwrite("dst.tif", dst)
    cv2.namedWindow("grayImg", cv2.WINDOW_NORMAL)
    cv2.namedWindow("dst", cv2.WINDOW_NORMAL)
    cv2.imshow("grayImg", img)
    cv2.imshow("dst", dst)
    cv2.waitKey()


def showDebugImage(images):
    imageTotal = len(images)
    for i in range(imageTotal):
        plt.subplot(1, imageTotal, i + 1)
        plt.title(images[i]["name"])
        plt.imshow(images[i]["img"], cmap="gray")
    plt.show()


frontMarkerRoi = (227, 338, 10, 60)              #(x, y, w, h)
rearMarker2Roi = (298, 324, 8, 54)

def getGradient():
    (x, y, w, h) = frontMarkerRoi
    lowThresh = 50
    highThresh = 100
    src = cv2.imread("./inputImgs/300.tif", cv2.IMREAD_GRAYSCALE)
    front_marker = src[int(y) : int(y) + int(h), int(x) : int(x) + int(w)]
    front_marker_edge = cv2.Canny(front_marker, lowThresh, highThresh)

    (x, y, w, h) = rearMarker2Roi
    rear_marker = src[int(y) : int(y) + int(h), int(x) : int(x) + int(w)]
    rear_marker_edge = cv2.Canny(rear_marker, lowThresh, highThresh)

    images = []
    singleImg = {"name": "frontMarker", "img": front_marker}
    images.append(singleImg)
    singleImg = {"name": "front_marker_edge", "img": front_marker_edge}
    images.append(singleImg)
    singleImg = {"name": "rear_marker", "img": rear_marker}
    images.append(singleImg)
    singleImg = {"name": "rear_marker_edge", "img": rear_marker_edge}
    images.append(singleImg)
    showDebugImage(images)


    # inputDir = './inputImgs/jpg'
    # for imgPath in glob.glob('{}/*.jpg'.format(inputDir)):
    #     print(imgPath)
    #     img = cv2.imread(imgPath)
    #     cropping = img[int(y) : int(y) + int(h), int(x) : int(x) + int(w)]
    #     grayImg = cv2.cvtColor(cropping, cv2.COLOR_BGR2GRAY)
    #     standartImg = grayImg / 255
    #     gradX = cv2.Sobel(standartImg, -1, 1, 0)
    #     gradY = cv2.Sobel(standartImg, -1, 0, 1)
    #     gradXY = cv2.addWeighted(gradX, 0.5, gradY, 0.5, 0)
    #     gradXY = cv2.convertScaleAbs(gradXY * 255)
    #     _, bin = cv2.threshold(grayImg, 0, 255, cv2.THRESH_OTSU)
    #     cannyImg = cv2.Canny(grayImg, lowThresh, highThresh)
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #     cannyImg = cv2.dilate(cannyImg, kernel)
    #     diff = marker_canny & ~cannyImg

    #     images = []
    #     singleImg = {"name": "cropColor", "img": cropping}
    #     images.append(singleImg)
    #     singleImg = {"name": "maker", "img": marker_canny}
    #     images.append(singleImg)
    #     singleImg = {"name": "bin", "img": bin}
    #     images.append(singleImg)
    #     singleImg = {"name": "diff", "img": diff}
    #     images.append(singleImg)
    #     singleImg = {"name": "cannyImg", "img": cannyImg}
    #     images.append(singleImg)
    #     showDebugImage(images)

        # plt.subplot(1, 5, 1)
        # plt.title("cropColor")
        # plt.imshow(cropping)
        # plt.subplot(1, 5, 2)
        # plt.title("maker")
        # plt.imshow(marker_canny, cmap='gray')
        # plt.subplot(1, 5, 3)
        # plt.title("bin")
        # plt.imshow(bin, cmap='gray')
        # plt.subplot(1, 5, 4)
        # plt.title("diff")
        # plt.imshow(diff, cmap='gray')
        # plt.subplot(1, 5, 5)
        # plt.title("cannyImg")
        # plt.imshow(cannyImg, cmap='gray')
        # plt.show()   
        # time.sleep(0.1) 

def getBinary(src):
    if len(src.shape) > 2:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    (x, y, w, h) = frontMarkerRoi
    front_marker = src[int(y) : int(y) + int(h), int(x) : int(x) + int(w)]
    _, front_marker_bin = cv2.threshold(front_marker, 0, 255, cv2.THRESH_OTSU)

    (x, y, w, h) = rearMarker2Roi
    rear_marker = src[int(y) : int(y) + int(h), int(x) : int(x) + int(w)]
    _, rear_marker_bin = cv2.threshold(rear_marker, 0, 255, cv2.THRESH_OTSU)

    images = []
    singleImg = {"name": "src", "img": src}
    images.append(singleImg)
    singleImg = {"name": "frontMarker", "img": front_marker}
    images.append(singleImg)
    singleImg = {"name": "frontBin", "img": front_marker_bin}
    images.append(singleImg)
    singleImg = {"name": "rearMarker", "img": rear_marker}
    images.append(singleImg)
    singleImg = {"name": "rearBin", "img": rear_marker_bin}
    images.append(singleImg)
    showDebugImage(images)


def saveFramesFromVideo():
    cap = cv2.VideoCapture('./inputVideos/44016F033D20210929224326017.avi')
    frameId = 0
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        if frameId == 300:
            cv2.imwrite('./inputImgs/' + str(frameId) + ".tif", frame)
            break
        frameId += 1


front_roi = (87, 420, 86, 178)
rear_roi = (174, 422, 165, 178)
def getEdgeWithVideo():
    inputDir = './inputVideos/day'
    exist = False
    frameId = 0
    for dir in glob.glob("{}/*.avi".format(inputDir)):  
        cap = cv2.VideoCapture(dir)
        while True:
            ret, img_color = cap.read()
            if ret is False:
                break
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            img_gray = np.rot90(img_gray, 3)
            img_gray = cv2.blur(img_gray, (11, 11))
            edge = cv2.Canny(img_gray, 40, 80)
            images = []
            singleImg = {"name": "src", "img": img_gray}
            images.append(singleImg)
            singleImg = {"name": "edge", "img": edge}
            images.append(singleImg)
            # showDebugImage(images)
            front_roi_img = edge[front_roi[1] : front_roi[1] + front_roi[3], front_roi[0] : front_roi[0] + front_roi[2]]
            rear_roi_img = edge[rear_roi[1] : rear_roi[1] + rear_roi[3], rear_roi[0] : rear_roi[0] + rear_roi[2]]
            front_count = cv2.countNonZero(front_roi_img)
            rear_count = cv2.countNonZero(rear_roi_img)
            print("front:", front_count, "      ", "rear:", rear_count)
            if exist is False and front_count > 10 and rear_count > 10:
                exist = True
                _, fileName = os.path.split(dir)
                cv2.imwrite(inputDir + "/result/" + fileName.split('.')[0] + "_" + str(frameId) + "_enter.jpg", img_color)
                print("车辆进场！")
            elif exist is True and front_count < 10 and rear_count < 10:
                exist = False
                _, fileName = os.path.split(dir)
                cv2.imwrite(inputDir + "/result/" + fileName.split('.')[0] + "_" + str(frameId) + "_exit.jpg", img_color)
                print("车辆出场！")


            cv2.imshow("src", img_gray)
            cv2.imshow("edge", edge)
            cv2.imshow("front", front_roi_img)
            cv2.imshow("rear", rear_roi_img)
            cv2.waitKey(20)
            frameId += 1




if __name__ == "__main__":

    # saveFramesFromVideo()
    # getGradient()
    getEdgeWithVideo()

    # cap = cv2.VideoCapture('./inputVideos/44016F033D20210929224326017.avi')
    # while True:
    #     ret, frame = cap.read()
    #     if ret is False:
    #         break
    #     getBinary(frame)


    # inputDir = 'imagesplicer_debug/20210508163835/result'
    # for dir in glob.glob("{}/*.jpg".format(inputDir)):  
    #     src = cv2.imread(dir)
    #     gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    #     ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_TRUNC)
    #     ret, threshOtsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    #     dst = equalizeHist(gray)
    #     claheDst = clahe(src)
    #     # cv2.namedWindow("src", cv2.WINDOW_NORMAL)
    #     # cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
    #     cv2.imshow("src", src)
    #     cv2.imshow("gray", gray)
    #     cv2.imshow("thresh", thresh)
    #     cv2.imshow("dst", dst)
    #     cv2.imshow("claheDst", claheDst)
    #     cv2.waitKey()

    # grayHist = calcGrayHist(img)
    # showGrayHist(grayHist)

    

    # adptive = cv2.adaptiveThreshold(gray, )

    # dst = gammaTrasfer(img, 3.0)

    # dst = logTransfer(gray, 100)

    # dst = SLT(img, 100, 160, 30, 230)

    # dst = equalizeHist(img)

    # dst1 = equalizeHist(gray)

    # cv2.imshow("dst", dst)
    # cv2.imshow("dst1", dst1)
    # cv2.waitKey()