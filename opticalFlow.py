import numpy as np
import cv2

"""
稀疏光流法
"""
def opticalLKFlow():
    videoDir = './test_video_dir/20210508162218.avi'
    cap = cv2.VideoCapture(videoDir)

    # ShiTomasi 角点检测参数
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

    # lucas kanade光流法参数
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # 创建随机颜色
    color = np.random.randint(0,255,(100,3))

    # 获取第一帧，找到角点
    ret, old_frame = cap.read()
    #找到原始灰度图
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    #获取图像中的角点，返回到p0中
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # 创建一个蒙版用来画轨迹
    mask = np.zeros_like(old_frame)

    while(1):
        ret,frame = cap.read()
        if ret is False:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算光流
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # 选取好的跟踪点
        good_new = p1[st==1]
        good_old = p0[st==1]

        # 画出轨迹
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)),(int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame,(int(a), int(b)),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)

        cv2.imshow('frame',img)
        if cv2.waitKey(30) & 0xff == ord('q'):
            break

        # 更新上一帧的图像和追踪点
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

    cv2.destroyAllWindows()
    cap.release()


'''
稠密光流法
'''
def calcOpticalFlowFarneback():
    videoDir = './test_video_dir/20210508155229.avi'
    cap = cv2.VideoCapture(videoDir)

    #获取第一帧
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)

    #遍历每一行的第1列
    hsv[...,1] = 255

    while(1):
        ret, frame2 = cap.read()
        if ret is False:
            break
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        #返回一个两通道的光流向量，实际上是每个点的像素位移值
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        #print(flow.shape)
        print(flow)

        #笛卡尔坐标转换为极坐标，获得极轴和极角
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        cv2.imshow('frame2',rgb)
        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',rgb)
        prvs = next

    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    # opticalLKFlow()
    calcOpticalFlowFarneback()