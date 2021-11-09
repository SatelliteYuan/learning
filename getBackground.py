from os import curdir
import cv2
import time
import copy

"""
OpenCV提供了一个称为BackgroundSubtractor的类，在分割前景和背景时很方便。
在OpenCV3中有三种背景分割器：K-Nearest（KNN）、Mixture of Gaussians（MOG2）、Geometric Multigid（GMG）
BackgroundSubtractor类是专门用于视频分析的，即BackgroundSubtractor类会对每帧的环境进行“学习”。
BackgroundSubtractor类常用来对不同帧进行比较，并存储以前的帧，可按时间推移方法来提高运动分析的结果。
"""

# MOG2背景分割器（Mixture of Gaussians）
def getBackgroundWithMOG2(videoDir):
    cap = cv2.VideoCapture(videoDir)
    mog = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.namedWindow("foreground", cv2.WINDOW_NORMAL)
    cv2.namedWindow("background", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        foregroundMask = mog.apply(frame)
        background = mog.getBackgroundImage()
        foregroundMask = cv2.morphologyEx(foregroundMask, cv2.MORPH_OPEN, kernel)
        cv2.imshow('frame', frame)
        cv2.imshow('foreground', foregroundMask)
        cv2.imshow('background', background)
        k = cv2.waitKey(1)
        if k == 27:
            cap.release()
            cv2.destroyAllWindows()
            break

# KNN背景分割器（K-Nearest)
def getBackgroundWittKNN(videoDir):
    cap = cv2.VideoCapture(videoDir)
    knn = cv2.createBackgroundSubtractorKNN()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.namedWindow("foreground", cv2.WINDOW_NORMAL)
    cv2.namedWindow("background", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        foregroundMask = knn.apply(frame)
        background = knn.getBackgroundImage()
        foregroundMask = cv2.morphologyEx(foregroundMask, cv2.MORPH_OPEN, kernel)
        cv2.imshow('frame', frame)
        cv2.imshow('foreground', foregroundMask)
        cv2.imshow('background', background)
        k = cv2.waitKey(1)
        if k == 27:
            cap.release()
            cv2.destroyAllWindows()
            break


def frameDiff(videoDir):
    cap = cv2.VideoCapture(videoDir)
    frameId = 0
    lastDiff = None
    preFrame = None
    curDiff = None
    cv2.namedWindow("foreground", cv2.WINDOW_NORMAL)
    cv2.namedWindow("lastDiff", cv2.WINDOW_NORMAL)
    cv2.namedWindow("curDiff", cv2.WINDOW_NORMAL)
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frameId == 0:
            preFrame = copy.copy(frame)
            frameId += 1
            continue
        elif frameId == 1:
            lastDiff = cv2.absdiff(preFrame, frame)
            ret, lastDiff = cv2.threshold(lastDiff, 0, 255, cv2.THRESH_OTSU)
        else:
            curDiff = cv2.absdiff(preFrame, frame)
            ret, curDiff = cv2.threshold(curDiff, 0, 255, cv2.THRESH_OTSU)
        preFrame = copy.copy(frame)
        if lastDiff is not None and curDiff is not None:
            foregound = cv2.bitwise_and(lastDiff, curDiff)
            cv2.imshow("foreground", foregound)
            cv2.imshow("curDiff", curDiff)
            cv2.imshow("lastDiff", lastDiff)
            cv2.imshow("frame", frame)
            cv2.waitKey(17)
        frameId += 1
        lastDiff = copy.copy(curDiff)


if __name__ == "__main__":
    videoDir = './test_video_dir/camera2_avi/20210508163423.avi'
    # getBackgroundWithMOG2(videoDir)
    # getBackgroundWittKNN(videoDir)
    frameDiff(videoDir)
