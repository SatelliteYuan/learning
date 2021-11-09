import cv2
from matplotlib import backend_bases
import matplotlib.pyplot as plt
import numpy as np
import glob
import time


inputDir = './test_video_dir/tooFast/44018B0D0420211019194637603.avi'
cap = cv2.VideoCapture(inputDir)

index = 0
preFrame = None
background = None

while True:
    ret, curFrame = cap.read()
    if ret is False:
        break
    curFrame = cv2.cvtColor(curFrame, cv2.COLOR_BGR2GRAY)
    curFrame = cv2.medianBlur(curFrame, 11)
    index += 1
    if index == 1:
        preFrame = curFrame.copy()
        continue
    diff = cv2.absdiff(curFrame, preFrame)
    ret, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (81, 81))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if background is None:
        background = (preFrame * 0.5 + curFrame * 0.5) * (~mask / 255)
    else:
        background = (background * 0.5 + curFrame * 0.5) * (~mask / 255)
    # bk = cv2.addWeighted(preFrame, 0.5, curFrame, 0.5, 0)
    # cv2.imshow("preFrame", preFrame)
    # cv2.imshow("curFrame", curFrame)
    # cv2.imshow("mask", mask)
    # cv2.imshow("background", background)
    # cv2.waitKey()
    preFrame = curFrame.copy() 
    # plt.subplot(1, 4, 1)
    # plt.title("preFrame")
    # plt.imshow(preFrame, cmap='gray')
    # plt.subplot(1, 4, 2)
    # plt.title("curFrame")
    # plt.imshow(img, cmap='gray')
    # plt.subplot(1, 4, 3)
    # plt.title("diff")
    # plt.imshow(diff, cmap='gray')
    # plt.subplot(1, 4, 4)
    # plt.title("bin")
    # plt.imshow(bin, cmap='gray')
    # plt.show()   
    # time.sleep(0.1)    
    
    # if cv2.waitKey() & 0xFF == ord(' '):
    #     continue

background = cv2.convertScaleAbs(background)
cv2.imshow("background", background)
cv2.waitKey()   



