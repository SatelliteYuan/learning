
import cv2
import os
import numpy as np

# def spliceImgs(imgs):


def matchKeypoint(kpsA, dpsA, kpsB, dpsB, ratio, reproThresh):
    # 第二步：实例化BFM匹配， 找出符合添加的关键点的索引
    bf = cv2.BFMatcher()

    matcher = bf.knnMatch(dpsA, dpsB, 2)

    matches = []

    for match in matcher:

        if len(match) == 2 and match[0].distance < match[1].distance * ratio:
            # 加入match[0]的索引
            matches.append((match[0].trainIdx, match[0].queryIdx))
    # 第三步：使用cv2.findHomography找出符合添加的H矩阵
    if len(matches) > 4:
        # 根据索引找出符合条件的位置
        kpsA = np.float32([kpsA[i] for (_, i) in matches])
        kpsB = np.float32([kpsB[i] for (i, _) in matches])
        (H, status) = cv2.findHomography(kpsA, kpsB, cv2.RANSAC, reproThresh)

        return (matches, H, status)
    return None


def detectandcompute(image):
    # 进行灰度值转化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # #实例化sift函数
    # sift = cv2.xfeatures2d.SIFT_create()
    # # 获得kps关键点和dps特征向量sift
    # kps, dps = sift.detectAndCompute(gray, None)
    orb = cv2.ORB_create()

    kps, dps = orb.detectAndCompute(gray, None)

    # 获得特征点的位置信息， 并转换数据类型
    kps = np.float32([kp.pt for kp in kps])

    return (kps, dps)


def showMatches(imageA, imageB, kpsA, kpsB, matches, status):
    # 将两个图像进行拼接
    # 根据图像的大小，构造全零矩阵
    via = np.zeros((max(imageB.shape[0], imageA.shape[0]),
                   imageA.shape[1] + imageB.shape[1], 3), np.uint8)
    # 将图像A和图像B放到全部都是零的图像中
    via[0:imageA.shape[0], 0:imageA.shape[1]] = imageA
    via[0:imageB.shape[0], imageA.shape[1]:] = imageB
    # 根据matches中的索引，构造出点的位置信息
    for (trainIdx, queryIdx), s in zip(matches, status):
        if s == 1:
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0] + imageA.shape[1]),
                   int(kpsB[trainIdx][1]))
            # 使用cv2.line进行画图操作
            cv2.line(via, ptA, ptB, (0, 255, 0), 1)

    return via

def sift(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转化为灰度图
    sift = cv2.SIFT_create()
    keyPoint, descriptor = sift.detectAndCompute(img, None)  # 特征提取得到关键点以及对应的描述符（特征向量）
    return img, keyPoint, descriptor

def surf(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转化为灰度图
    surf = cv2.SURF_create()
    keyPoint, descriptor = surf.detectAndCompute(img, None)  # 特征提取得到关键点以及对应的描述符（特征向量）
    return img, keyPoint, descriptor

def orb(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转化为灰度图
    orb = cv2.ORB_create()
    keyPoint, descriptor = orb.detectAndCompute(img, None)  # 特征提取得到关键点以及对应的描述符（特征向量）
    return img, keyPoint, descriptor

def filterMatcher(matches, kp1, kp2):
    filteredMatches = []
    for m in matches:
        for n in matches:
            if(m != n and m.distance >= n.distance*0.75):
                matches.remove(m)
                break

    for m in matches:
        if(abs(kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0]) > 10 and abs(kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1]) < 30):
            filteredMatches.append(m)
    
    # for m in filteredMatches:
    #     print(kp1[m.queryIdx].pt[0], "^^^^^", kp2[m.trainIdx].pt[0])
    #     print(kp1[m.queryIdx].pt[1], "^^^^^", kp2[m.trainIdx].pt[1], '\n')

    return filteredMatches

def match(imageA, imageB, method):
    if(method == 'sift'):
        img1, kp1, des1 = sift(imageA)
        img2, kp2, des2 = sift(imageB)
        # sift的normType应该使用NORM_L2或者NORM_L1
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        knnMatches = bf.knnMatch(des1, des2, k=1)  # drawMatchesKnn
    elif (method == 'surf'):
        img1, kp1, des1 = surf(imageA)
        img2, kp2, des2 = surf(imageB)
        # surf的normType应该使用NORM_L2或者NORM_L1
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        knnMatches = bf.knnMatch(des1, des2, k=1)  # drawMatchesKnn
    elif(method == 'orb'):
        img1, kp1, des1 = orb(imageA)
        img2, kp2, des2 = orb(imageB)
        # orb的normType应该使用NORM_HAMMING
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        knnMatches = bf.knnMatch(des1, des2, k=1)  # drawMatchesKnn

    # 过滤
    fileteredMatches = filterMatcher(matches, kp1, kp2)

    img = cv2.drawMatches(img1, kp1, img2, kp2, fileteredMatches, img2, flags=2)
    cv2.imshow(method, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def testSpliceImgs():
    import glob
    grayImages = []
    colorImages = []
    for imgPath in glob.glob('inputImgs/*.jpg'):
        gray = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        color = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        if gray is None or color is None:
            continue
        grayImages.append(gray)
        colorImages.append(color)
    
    diff = cv2.absdiff(grayImages[0], grayImages[1])
    ret, binary = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    kernel = np.ones((11, 11), np.uint8)
    close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    open = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    binary = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel)

    images = []
    mask = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    # cv2.imshow("mask", mask)
    # cv2.waitKey()
    for img in colorImages:
        img = cv2.bitwise_and(img, mask)
        images.append(img)

    [imageB, imageA] = colorImages
    # # 第一步：计算kpsA和dpsA
    # (kpsA, dpsA) = detectandcompute(imageA)
    # (kpsB, dpsB) = detectandcompute(imageB)
    # # 获得变化的矩阵H
    # ratio = 0.75
    # reproThresh = 2
    # M = matchKeypoint(kpsA, dpsA, kpsB, dpsB, ratio, reproThresh)

    # if M is None:
    #     return None
    # (matches, H, status) = M

    # # 第四步：使用cv2.warpPerspective获得经过H变化后的图像
    # result = cv2.warpPerspective(
    #     imageA, H, (imageA.shape[1] + imageB.shape[1], imageB.shape[0]))

    # # 第五步：将图像B填充到进过H变化后的图像，获得最终的图像
    # result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

    # via = showMatches(imageA, imageB, kpsA, kpsB, matches, status)
    # cv2.imshow("via", via)
    # cv2.imshow("result", result)
    # cv2.waitKey()
    # for i, img in enumerate(images):
    #     cv2.imshow(f"img{i}", img)
    # cv2.waitKey()

    # stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    # status, stitched = stitcher.stitch(images)
    # ret, stitched = stitcher.stitch(images[0], images[1])
    # if ret != cv2.Stitcher_OK:
    #     print("stitch fail!")
    # else:
    #     cv2.imshow(f"img", stitched)
    #     cv2.waitKey()

    # match(imageA, imageB, 'sift')

    methods = ['sift','orb']
    for method in methods:
        match(imageA, imageB, method)

    return

def spliceTwoImages():
    inputDir = "./imagesplicer_debug/20210508163835/result/"
    imageA = cv2.imread(inputDir + "14.jpg")
    imageB = cv2.imread(inputDir + "15.jpg")

    methods = ['sift','orb']
    for method in methods:
        match(imageA, imageB, method)


if __name__ == '__main__':
    if os.path.isdir('./test_output') is False:
        os.makedirs('./test_output')

    spliceTwoImages()
    # testSpliceImgs()