import cv2
import time
import os
import numpy as np
import sys
import glob

from numpy.lib.function_base import rot90


# def calibrate_single(imgNums, CheckerboardSize, Nx_cor, Ny_cor, saveFile=False, saveImages=False):
#     '''
#     单目(普通+广角/鱼眼)摄像头标定
#     :param imgNums: 标定所需样本数,一般在20~40之间.按键盘空格键实时拍摄
#     :param CheckerboardSize: 标定的棋盘格尺寸,必须为整数.(单位:mm或0.1mm)
#     :param Nx_cor: 棋盘格横向内角数
#     :param Ny_cor: 棋盘格纵向内角数
#     :param saveFile: 是否保存标定结果,默认不保存.
#     :param saveImages: 是否保存图片,默认不保存.
#     :return mtx: 内参数矩阵.{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}
#     :return dist: 畸变系数.(k_1,k_2,p_1,p_2,k_3)
#     '''
#     # 找棋盘格角点(角点精准化迭代过程的终止条件)
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, CheckerboardSize, 1e-6)  # (3,27,1e-6)
#     flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE  # 11
#     flags_fisheye = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW  # 14

#     # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
#     objp = np.zeros((1, Nx_cor * Ny_cor, 3), np.float32)
#     objp[0, :, :2] = np.mgrid[0:Nx_cor, 0:Ny_cor].T.reshape(-1, 2)

#     # 储存棋盘格角点的世界坐标和图像坐标对
#     objpoints = []  # 在世界坐标系中的三维点
#     imgpoints = []  # 在图像平面的二维点

#     count = 0  # 用来标志成功检测到的棋盘格画面数量

#     while (True):
#         ret, frame = cap.read()
#         frame = cv2.resize(frame, imageSize)
#         cv2.imshow('frame', frame)

#         if cv2.waitKey(1) & 0xFF == ord(' '):
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             # 寻找棋盘格模板的角点
#             ok, corners = cv2.findChessboardCorners(gray, (Nx_cor, Ny_cor), flags)
#             if count >= imgNums:
#                 break
#             if ok:  # 如果找到，添加目标点，图像点
#                 objpoints.append(objp)
#                 cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 获取更精确的角点位置
#                 imgpoints.append(corners)

#                 # 将角点在图像上显示
#                 cv2.drawChessboardCorners(frame, (Nx_cor, Ny_cor), corners, ok)
#                 count += 1
#                 if saveImages:
#                     cv2.imwrite('../imgs/' + str(count) + '.jpg', frame)
#                 print('NO.' + str(count))

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     global mtx, dist

#     # 标定. rvec和tvec是在获取了相机内参mtx,dist之后通过内部调用solvePnPRansac()函数获得的
#     # ret为标定结果，mtx为内参数矩阵，dist为畸变系数，rvecs为旋转矩阵，tvecs为平移向量
#     ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
#         objpoints, imgpoints, gray.shape[:2][::-1], None, criteria
#     )
#     # 摄像头内参mtx = [[f_x,0,c_x][0,f_y,c_y][0,0,1]]
#     print('mtx=np.array( ' + str(mtx.tolist()) + " )")
#     # 畸变系数dist = (k1,k2,p1,p2,k3)
#     print('dist=np.array( ' + str(dist.tolist()) + " )")

#     # 鱼眼/大广角镜头的单目标定
#     K = np.zeros((3, 3))
#     D = np.zeros((4, 1))
#     RR = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objpoints))]
#     TT = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objpoints))]
#     rms, _, _, _, _ = cv2.fisheye.calibrate(
#         objpoints, imgpoints, gray.shape[:2][::-1], K, D, RR, TT, flags_fisheye, criteria
#     )
#     # 摄像头内参,此结果与mtx相比更为稳定和精确
#     print("K=np.array( " + str(K.tolist()) + " )")
#     # 畸变系数D = (k1,k2,k3,k4)
#     print("D=np.array( " + str(D.tolist()) + " )")
#     # 计算反投影误差
#     mean_error = 0
#     for i in range(len(objpoints)):
#         imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#         error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
#         mean_error += error
#     print("total error: ", mean_error / len(objpoints))

#     if saveFile:
#         np.savez("./calibrate.npz", mtx=mtx, dist=dist, K=K, D=D)
#     cv2.destroyAllWindows()
#     return mtx, dist, K, D



# def undistort(img_path, balance=0.0, dim2=None, dim3=None):
#     img = cv2.imread(img_path)
#     dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
#     assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
#     if not dim2:
#         dim2 = dim1
#     if not dim3:
#         dim3 = dim1
#     scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
#     scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
#     # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
#     new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
#     map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
#     undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
#     cv2.imshow("undistorted", undistorted_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# DIM = None
# K = None
# D = None

def saveChessImages():
    frameId = 0
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        # cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            frameId += 1
            print("capture blank imformation!", frameId)
            flipImg = np.rot90(frame)
            cv2.imwrite("./imgs/" + str(frameId) + "_frame.tif", flipImg)
            continue
    cap.release()
    cv2.destroyAllWindows()


def undistortion(img, mtx, dist):
    h, w = img.shape[:2]
    #getOptimalNewCameraMatrix调节视场大小，为1时视场大小不变，小于1时缩放视场
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    if roi != (0, 0, 0, 0):
        dst = dst[y:y + h, x:x + w]

    return dst


def calibrateCamera():
    inputDir = "./imgs"
    (Colums, Rows) = (6, 9)    #水平和竖直方向上角点个数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((Colums * Rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:Colums, 0:Rows].T.reshape(-1, 2)
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    for imgPath in glob.glob("{}/*tif".format(inputDir)):
        src = cv2.imread(imgPath, cv2.IMREAD_ANYCOLOR)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (Colums, Rows), None)
        if ret is True:
            subpixCorners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(subpixCorners)
            # cv2.drawChessboardCorners(src, (Rows, Colums), subpixCorners, ret)
            # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            # cv2.imshow("img", src)
            # cv2.waitKey(0)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(mtx, dist)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        print ("error{}: ".format(str(i)), error)
        mean_error += error

    print ("total error: ", mean_error / len(objpoints))

    return mtx, dist


if __name__ == '__main__':
    # saveChessImages()
    mtx, dist = calibrateCamera()

    testDir = "./vehicleHeadImages"
    for dir in glob.glob("{}/*.jpg".format(testDir)):
        img = cv2.imread(dir)
        # dst = undistortion(img, mtx, dist)
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        x, y, w, h = roi
        if roi != (0, 0, 0, 0):
            crop = dst[y:y + h, x:x + w]
            # cv2.imshow("crop", crop)
        start = time.time()
        dstNoneCrop = cv2.undistort(img, mtx, dist)
        print("time:", time.time() - start)

        # outputDir = testDir + "/result"
        # if os.path.isdir(outputDir) is False:
        #     os.path.mkdir(outputDir)
        # cv2.imwrite(os.path.join(outputDir, os.path.basename(dir)), dstNoneCrop)

        cv2.namedWindow("src", cv2.WINDOW_NORMAL)
        cv2.namedWindow("dstNoneCrop", cv2.WINDOW_NORMAL)
        cv2.namedWindow("dst", cv2.WINDOW_NORMAL)
        cv2.imshow("src", img)
        cv2.imshow("dst", dst)
        cv2.imshow("dstNoneCrop", dstNoneCrop)
        cv2.waitKey()

    # frameId = 0
    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # # print('height=', height, 'width=', width)
    # while True:
    #     ret, frame = cap.read()
    #     if ret is False:
    #         break
    #     flipImg = np.rot90(frame)
    #     # cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    #     cv2.imshow("Frame", flipImg)

    #     if cv2.waitKey(1) & 0xFF == ord(' '):
    #         frameId += 1
    #         print("capture blank imformation!", frameId)
    #         # flipImg = np.rot90(frame)
    #         # cv2.namedWindow("flipImg", cv2.WINDOW_NORMAL)
    #         dst = cv2.undistort(flipImg, mtx, dist)
    #         cv2.imshow("dst", dst)
    #         continue
    # cap.release()
    # cv2.destroyAllWindows()



