import cv2
import numpy as np
import control_bar_hsv


roi = (0, 456, 480, 184)    #阴影检测区域  (x, y, w, h)

def eliminate_shadow(img_bk, img_cur):
    x, y, w, h = roi
    img_bk_roi = img_bk[int(y) : int(y) + int(h), int(x) : int(x) + int(w)]
    img_cur_roi = img_cur[int(y) : int(y) + int(h), int(x) : int(x) + int(w)]
    img_diff = cv2.absdiff(img_bk_roi, img_cur_roi)
    cv2.imshow("bk", img_bk_roi)
    cv2.imshow("cur", img_cur_roi)
    cv2.imshow("diff", img_diff)
    # cv2.waitKey(10)
    img_bk_roi_hsv = cv2.cvtColor(img_bk_roi, cv2.COLOR_BGR2HSV)
    img_cur_roi_hsv = cv2.cvtColor(img_cur_roi, cv2.COLOR_BGR2HSV)
    img_diff_hsv = cv2.absdiff(img_bk_roi_hsv, img_cur_roi_hsv)
    h,s, v = cv2.split(img_diff_hsv)
    cv2.imshow("h", h)
    cv2.imshow("s", s)
    cv2.imshow("v", v)
    cv2.waitKey(0)
    # obj = control_bar_hsv.ControlBarHSV(img_diff_hsv)
    # obj.run()


if __name__ == "__main__":
    img_bk = cv2.imread('./inputImgs/BK.tif', cv2.IMREAD_COLOR)
    img_cur = cv2.imread('./inputImgs/I_2.tif', cv2.IMREAD_COLOR)
    eliminate_shadow(img_bk, img_cur)