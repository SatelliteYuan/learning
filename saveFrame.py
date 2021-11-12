import cv2
import os
import numpy as np


def save_frame_of_video():
    video_name = './inputVideos/1110.avi'
    dir_output = './inputImgs/frames/'
    if not os.path.isdir(dir_output):
        os.mkdir(dir_output)

    cap = cv2.VideoCapture(video_name)
    frameId = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = np.rot90(frame, 3)
        cv2.imwrite(dir_output + str(frameId) + "_frame.tif", frame)
        frameId += 1



if __name__ == "__main__":
    save_frame_of_video()