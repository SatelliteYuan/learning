import numpy as np
import cv2
import os
import copy
import glob
import time
import logging
import queue


# 显示中间过程图
def show_debug_image(images):
    pass
    # import matplotlib.pyplot as plt
    # imageTotal = len(images)
    # for i in range(imageTotal):
    #     plt.subplot(1, imageTotal, i + 1)
    #     plt.title(images[i]["name"])
    #     plt.imshow(images[i]["img"], cmap="gray")
    # plt.show()

#生成时间戳用作保存结果名
def getTimeString():
    times = time.localtime()
    time_str = (str(times.tm_year) + str(times.tm_mon).zfill(2) + str(times.tm_mday).zfill(2) + 
        str(times.tm_hour).zfill(2) + str(times.tm_min).zfill(2))
    return time_str


class Marker(object):
    def __init__(self, template_mask_path, template_edge_path, marker_roi):
        self.template_mask = None
        self.template_edge = None
        self.template_mask_path = template_mask_path
        self.template_edge_path = template_edge_path
        self.marker_roi = marker_roi
        self.mask = None
        self.edge = None
        self.lower_thresh = 0
        self.upper_thresh = 0
        self.binary_thresh = 0
        self.load_marker_image()
        self.update_marker_threshold()

    # 加载marker模板图
    def load_marker_image(self):
        if self.template_mask_path is not None:
            self.template_mask = cv2.imread(self.template_mask_path, cv2.IMREAD_GRAYSCALE)

        if self.template_edge_path is not None:
            self.template_edge = cv2.imread(self.template_edge_path, cv2.IMREAD_GRAYSCALE)

    # 更新判断marker是否被遮挡相关的阈值
    def update_marker_threshold(self):
        if self.template_edge is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilate_edge = cv2.dilate(self.template_edge, kernel)
            lower_limit = cv2.countNonZero(self.template_edge) * 0.1
            upper_limit = cv2.countNonZero(dilate_edge)
            total_pixel = self.template_edge.shape[0] * self.template_edge.shape[1]
            upper_limit = upper_limit + (total_pixel - upper_limit) * 0.5
            self.lower_thresh = lower_limit
            self.upper_thresh = upper_limit

        if self.template_mask is not None:
            self.binary_thresh = int(int(self.marker_roi[2]) * int(self.marker_roi[3]) * 0.15)


class VehicleDetection(object):
    def __init__(self, front_marker_roi, rear_marker_roi):
        # marker模板
        self.front_marker_roi = front_marker_roi  # (x, y, w, h)
        self.rear_marker_roi = rear_marker_roi
        self.front_marker_mask_path = './frontMarkerMask.tif'
        self.front_marker_edge_path = './frontMarkerEdge.tif'
        self.rear_marker_mask_path = './rearMarkerMask.tif'
        self.rear_marker_edge_path = './rearMarkerEdge.tif'
        self.front_marker = Marker(self.front_marker_mask_path, self.front_marker_edge_path, self.front_marker_roi)
        self.rear_marker = Marker(self.rear_marker_mask_path, self.rear_marker_edge_path, self.rear_marker_roi)

        # 状态记录
        self.frame_index = 0
        self.enter_frame_index = 0
        self.leave_frame_index = 0
        self.vehicle_exist = False

        # 通用设置
        self.frame_interval = 1  # 处理帧间隔
        self.distraction_save_path = "./distraction"
        self.enter_and_leave_save_path = "./result"
        self.with_gradient = True
        self.show_debug_image = False
        self.whether_rotate = False
        self.cur_time = getTimeString()
        self.make_result_dir()

        # 用于更新背景
        self.update_template_enable = True
        self.update_template_interval = 30 * 60 * 30  # 更新一次模板间隔的帧数
        self.frame_count = 0  # 间隔计数
        self.save_distraction = False
        self.collect_markers = False
        self.front_marker_queue = queue.Queue(maxsize=5)
        self.rear_marker_queue = queue.Queue(maxsize=5)

    # 生成用于保存结果图的文件夹
    def make_result_dir(self):
        if os.path.isdir(self.enter_and_leave_save_path) is False:
            os.mkdir(self.enter_and_leave_save_path)

        if os.path.isdir(self.distraction_save_path) is False:
            os.mkdir(self.distraction_save_path)

    # 基于marker的梯度信息来判断marker是否被遮挡
    def whether_shield_with_gradient(self, marker: Marker, cropped):
        if len(cropped.shape) == 3:
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        canny_img = cv2.Canny(cropped, 30, 70)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        canny_img = cv2.dilate(canny_img, kernel)
        diff = marker.template_edge & ~canny_img

        if self.show_debug_image is True:
            images = []
            single_img = {"name": "src", "img": cropped}
            images.append(single_img)
            single_img = {"name": "markerEdge", "img": marker.template_edge}
            images.append(single_img)
            single_img = {"name": "curEdge", "img": canny_img}
            images.append(single_img)
            single_img = {"name": "diff", "img": diff}
            images.append(single_img)
            show_debug_image(images)

        count_diff = cv2.countNonZero(diff)
        count_cur = cv2.countNonZero(canny_img)
        if count_diff < marker.lower_thresh and count_cur < marker.upper_thresh:
            return False
        else:
            return True

    # 基于marker的二值化图来判断marker是否被遮挡
    def whether_shield_with_thresh(self, marker: Marker, cropped):
        if len(cropped.shape) == 3:
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        ret, binary = cv2.threshold(cropped, 0, 255, cv2.THRESH_OTSU)
        if ret is False:
            logging.error("threshold失败！")
            return False
        diff = cv2.absdiff(binary, marker.template_mask)

        if self.show_debug_image is True:
            images = []
            singleImg = {"name": "src", "img": cropped}
            images.append(singleImg)
            singleImg = {"name": "marker", "img": marker.template_mask}
            images.append(singleImg)
            singleImg = {"name": "cur", "img": binary}
            images.append(singleImg)
            singleImg = {"name": "diff", "img": diff}
            images.append(singleImg)
            show_debug_image(images)

        none_zero = cv2.countNonZero(diff)
        if none_zero > marker.binary_thresh:
            return True
        else:
            return False

    # 更新marker模板
    def update_marker_template(self):
        if not self.front_marker_queue.full() or not self.rear_marker_queue.full():
            logging.error("未能收集足够的marker背景，更新背景失败！")
            return False

        scale = 1 / self.front_marker_queue.qsize()
        front_marker = None
        while not self.front_marker_queue.empty():
            cur_marker = self.front_marker_queue.get()
            if cur_marker is None:
                logging.error("收集到marker背景为空，更新背景失败！")
                return False
            if front_marker is None:
                front_marker = cur_marker * scale
            else:
                front_marker += cur_marker * scale
        binary_thresh = int(255 * scale * 2) + 1  # 背景图组中同一位置出现两次以上边缘点则标记入背景
        _, self.front_marker.template_edge = cv2.threshold(front_marker, binary_thresh, 255, cv2.THRESH_BINARY)
        self.front_marker.template_edge = self.front_marker.template_edge.astype(np.uint8)

        scale = 1 / self.rear_marker_queue.qsize()
        rear_marker = None
        while not self.rear_marker_queue.empty():
            cur_marker = self.rear_marker_queue.get()
            if cur_marker is None:
                logging.error("收集到marker背景为空，更新背景失败！")
                return False
            if rear_marker is None:
                rear_marker = cur_marker * scale
            else:
                rear_marker += cur_marker * scale
        binary_thresh = int(255 * scale * 2) + 1
        _, self.rear_marker.template_edge = cv2.threshold(rear_marker, binary_thresh, 255, cv2.THRESH_BINARY)
        self.rear_marker.template_edge = self.rear_marker.template_edge.astype(np.uint8)

        self.front_marker.update_marker_threshold()
        self.rear_marker.update_marker_threshold()
        logging.debug("成功更新marker背景！")

        return True

    # 收集用于叠加生成背景的marker图
    def collect_marker_background(self, img_src, marker_position):
        if marker_position == 'front':
            marker_edge = self.get_marker_edge(img_src)
            if self.front_marker_queue.full():
                self.front_marker_queue.get()
            self.front_marker_queue.put(marker_edge)
        elif marker_position == 'rear':
            marker_edge = self.get_marker_edge(img_src)
            if self.rear_marker_queue.full():
                self.rear_marker_queue.get()
            self.rear_marker_queue.put(marker_edge)

    # 检测车辆是否位于检测区域内
    def check_vehicle_exist(self, frame):
        if (self.front_marker.template_edge is None or self.front_marker.template_mask is None 
                or self.rear_marker.template_edge is None or self.rear_marker.template_mask is None):
            logging.error('marker模板为空！')
            return False

        if self.whether_rotate is True:
            frame = np.rot90(frame, 3)

        # 检测前参照物遮挡状态
        self.frame_index += 1
        self.frame_count += 1
        (x, y, w, h) = self.front_marker_roi
        front_crop = frame[int(y): int(y) + int(h), int(x): int(x) + int(w)]
        (x, y, w, h) = self.rear_marker_roi
        rear_crop = frame[int(y): int(y) + int(h), int(x): int(x) + int(w)]
        if self.with_gradient is True:
            whether_shield_flag = self.whether_shield_with_gradient(self.front_marker, front_crop)
        else:
            whether_shield_flag = self.whether_shield_with_thresh(self.front_marker, front_crop)

        # 车辆进入检测区域
        if whether_shield_flag is True and self.vehicle_exist is False:
            if abs(self.frame_index - self.leave_frame_index) < 5:
                return self.vehicle_exist
            if self.with_gradient is True:
                review_flag = self.whether_shield_with_gradient(self.rear_marker, rear_crop)
            else:
                review_flag = self.whether_shield_with_thresh(self.rear_marker, rear_crop)
            if review_flag is True:
                self.vehicle_exist = True
                self.enter_frame_index = self.frame_index
                cv2.imwrite(os.path.join(self.enter_and_leave_save_path, 
                    '{}_{}_enterFrame.jpg'.format(self.cur_time, str(self.frame_index))), frame)
                self.save_distraction = True
                self.collect_markers = True
        elif whether_shield_flag is False and self.vehicle_exist is True:
            if abs(self.frame_index - self.enter_frame_index) < 5:  # 离开帧和进入帧相邻则认为是错误匹配
                return self.vehicle_exist
            if self.with_gradient is True:
                review_flag = self.whether_shield_with_gradient(self.rear_marker, rear_crop)
            else:
                review_flag = self.whether_shield_with_thresh(self.rear_marker,
                                                          rear_crop)  # 判断后marker是否被遮挡，只有两个marker都能找到才确定车辆离开
            if review_flag is True:
                if self.save_distraction is True:  # 同一辆车中间部分只保存一张图片
                    logging.debug("检测到拖车中间部分！")
                    cv2.imwrite(os.path.join(self.distraction_save_path, 
                        '{}_{}_distractionFrame.jpg'.format(self.cur_time, str(self.frame_index))), frame)
                    self.save_distraction = False
            else:
                self.vehicle_exist = False
                self.leave_frame_index = self.frame_index
                cv2.imwrite(os.path.join(self.enter_and_leave_save_path, 
                    '{}_{}_leaveFrame.jpg'.format(self.cur_time, str(self.frame_index))), frame)
            
        #更新图片保存序号
        if self.frame_index > 5000:
            self.frame_index = 0
            self.cur_time = getTimeString()

        # 更新背景
        if self.update_template_enable is True:
            if self.vehicle_exist is False and abs(
                    self.frame_index - self.leave_frame_index) > 2 and self.collect_markers is True:
                self.collect_marker_background(front_crop, "front")
                self.collect_marker_background(rear_crop, "rear")
                self.collect_markers = False

            if self.frame_count >= self.update_template_interval:
                if self.update_marker_template():
                    self.frame_count = 0

        return self.vehicle_exist

    # 生成对应marker的模板图，首次使用时需要手动调用
    def generate_marker_template(self, background_img, force=False):
        if os.path.exists(os.path.dirname(self.front_marker_edge_path)) is False:
            os.makedirs(os.path.dirname(self.front_marker_edge_path), exist_ok=True)
            
        if force or self.front_marker.template_edge is None:
            self.front_marker.template_edge = self.get_marker_edge(background_img, self.front_marker_roi)
            if self.front_marker.template_edge is not None:
                cv2.imwrite(self.front_marker_edge_path, self.front_marker.template_edge)
        if force or self.front_marker.template_mask is None:
            self.front_marker.template_mask = self.get_marker_mask(background_img, self.front_marker_roi)
            if self.front_marker.template_mask is not None:
                cv2.imwrite(self.front_marker_mask_path, self.front_marker.template_mask)
        self.front_marker.update_marker_threshold()

        if force or self.rear_marker.template_edge is None:
            self.rear_marker.template_edge = self.get_marker_edge(background_img, self.rear_marker_roi)
            if self.rear_marker.template_edge is not None:
                cv2.imwrite(self.rear_marker_edge_path, self.rear_marker.template_edge)

        if force or self.rear_marker.template_mask is None:
            self.rear_marker.template_mask = self.get_marker_mask(background_img, self.rear_marker_roi)
            if self.rear_marker.template_mask is not None:
                cv2.imwrite(self.rear_marker_mask_path, self.rear_marker.template_mask)
        self.rear_marker.update_marker_threshold()

    # 获取marker的二值化mask图
    def get_marker_mask(self, img, roi):
        if len(img.shape) > 2:
            cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (x, y, w, h) = roi
        cropped = cvt_img[int(y): int(y) + int(h), int(x): int(x) + int(w)]
        _, mask = cv2.threshold(cropped, 0, 255, cv2.THRESH_OTSU)

        if self.show_debug_image is True:
            images = []
            single_img = {"name": "img", "img": cvt_img}
            images.append(single_img)
            single_img = {"name": "cropped", "img": cropped}
            images.append(single_img)
            single_img = {"name": "mask", "img": mask}
            images.append(single_img)
            show_debug_image(images)

        return mask

    # 获取marker的边缘图
    def get_marker_edge(self, img, roi=None):
        if len(img.shape) > 2:
            cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if roi is not None:
            (x, y, w, h) = roi
            cropped = cvt_img[int(y): int(y) + int(h), int(x): int(x) + int(w)]
        else:
            cropped = copy.copy(img)
        edge_img = cv2.Canny(cropped, 50, 100)

        if self.show_debug_image is True:
            images = []
            single_img = {"name": "img", "img": cvt_img}
            images.append(single_img)
            single_img = {"name": "cropped", "img": cropped}
            images.append(single_img)
            single_img = {"name": "edge", "img": edge_img}
            images.append(single_img)
            show_debug_image(images)

        return edge_img


# 运行图片集测试
def test_with_images(front_marker_roi, rear_marker_roi):
    detect_obj = VehicleDetection(front_marker_roi, rear_marker_roi)
    input_dir = "./inputImgs/xilong1"
    detect_obj.whether_rotate = True
    for dir in glob.glob("{}/*.jpg".format(input_dir)):
        logging.debug(dir)
        frame = cv2.imread(dir, cv2.IMREAD_COLOR)
        detect_obj.check_vehicle_exist(frame)


# 运行本地视频测试
def test_with_video(front_marker_roi, rear_marker_roi):
    detect_obj = VehicleDetection(front_marker_roi, rear_marker_roi)
    input_dir = "./inputVideos/day/1110-1.avi"
    cap = cv2.VideoCapture(input_dir)
    detect_obj.whether_rotate = True

    generated_marker_template = True

    while True:
        ret, frame = cap.read()
        if ret is False:
            break

        if generated_marker_template is False:
            detect_obj.generate_marker_template(frame, True)
            generated_marker_template = True
            continue

        detect_obj.check_vehicle_exist(frame)


# 运行摄像头测试
def test_with_camera(front_marker_roi, rear_marker_roi):
    detect_obj = VehicleDetection(front_marker_roi, rear_marker_roi)
    detect_obj.whether_rotate = True
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        detect_obj.check_vehicle_exist(frame)


if __name__ == '__main__':
    # 初始化logging
    from logging import handlers

    logging.basicConfig(level=logging.DEBUG)
    th = handlers.TimedRotatingFileHandler(filename='all.log', when='D', backupCount=3, encoding='utf-8')
    th.setFormatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    logging.getLogger().addHandler(th)

    # XiLong1 resizeRatio = 1.0
    # front_marker_roi = (264, 418, 48, 22)
    # rear_marker_roi = (344, 400, 53, 23)

    # XiLong2 resizeRatio = 1.0
    # front_marker_roi = (199, 398, 63, 22)
    # rear_marker_roi = (147, 377, 38, 38)

    # XiLong2 resizeRatio = 0.5
    front_marker_roi = (130, 383, 55, 18)
    rear_marker_roi = (217, 368, 39, 26)

    # test_with_images(front_marker_roi, rear_marker_roi)
    # test_with_video(front_marker_roi, rear_marker_roi)
    test_with_camera(front_marker_roi, rear_marker_roi)

    logging.debug('Test finish!')
