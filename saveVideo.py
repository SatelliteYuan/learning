import cv2
import sys
import numpy as np
import logging
from logging import handlers

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


def saveVideo(outputName, frameTotal):
    cap = cv2.VideoCapture(0)
    ret = cap.isOpened()
    if ret is False:
        log.logger.error("open camera fail!")
        exit()
   
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(outputName, 
                                  cv2.VideoWriter_fourcc(*'XVID'), 
                                  cap.get(cv2.CAP_PROP_FPS), 
                                  size)
    frameId = 0
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        frame = np.rot90(frame, 3)
        videoWriter.write(frame)
        if frameId > int(frameTotal):
            break
        frameId += 1

    cap.release()
    log.logger.debug("视频已保存完!")

    

log = Logger('saveVideo.log',level='debug')


"""
param1:  保存的视频名
param2:  收集视频的帧数
"""
if __name__ == "__main__":
    log.logger.debug(len(sys.argv))
    if len(sys.argv) < 3:
        log.logger.error("请输入参数！ videoName, frameTotal")
        exit()
    saveVideo(sys.argv[1], sys.argv[2])
    exit()