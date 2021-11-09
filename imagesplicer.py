"""
Author       : xujiawei
Date         : 2021-03-12 10:55:29
Description  : 图片拼接类
FilePath     : /VehTypeCameraServer/trunk/splicer/imagesplicer.py
"""
import copy
import logging
import os
import datetime
from collections import deque

import cv2
import numpy as np
from numpy.core.records import array
from tqdm import tqdm

from common.image_operator import *
from common import vehicle_info_cache


log_offset_details = False


def debug_print(self, *args):
    if log_offset_details:
        print(self, args)


class OffSetResult:
    def __init__(self, offset, could_append_to_last_speed=True):
        self.offset = offset
        self.could_append_to_last_speed = could_append_to_last_speed


class ImageSplicer(object):
    """图片拼接类

    Args:
        height (int): 画面高度
        width (int): 画面宽度
        detect_vehicle_view (tuple(int, int, int, int)): 检测车辆的截选区域，(top,bottom,left,right)
        calc_offset_view (tuple(int, int, int, int)): 计算位移的截选区域，(top,bottom,left,right)
        min_detect_contour_size (int): 检测车辆的最小轮廓大小
    """

    def __init__(self, height, width, detect_vehicle_view, calc_offset_view, splice_x_pos, scale=1.0,
                 reverse_lane=False, min_detect_contour_size=150, detect_car_leave_by_video=False):
        self.height = int(height)
        self.width = int(width)
        self.below_detect_vehicle_view = detect_vehicle_view
        self.calc_offset_view = calc_offset_view

        # 上检测框
        self.detect_below_vehicle_view_crop_op = Crop(
            view=self.below_detect_vehicle_view)
        # 下检测框, 下检测框左右位置与下检测框相同, 但上下位置写死
        self.above_detect_vehicle_view = [
            0, 100, self.below_detect_vehicle_view[2], self.below_detect_vehicle_view[3]]
        self.detect_above_vehicle_view_crop_op = Crop(
            view=self.above_detect_vehicle_view)

        self.calc_offset_view_crop_op = Crop(view=self.calc_offset_view)
        self.splice_x_pos = splice_x_pos
        self.frame_scale = float(scale)

        # 最小的拼接帧数, 当得到的有效帧数小于该值时, 不进行拼接, 用于过滤倒车
        self.min_splicer_frame_count = 4

        # 当位移小于该值时, 忽略当帧, 直到位移大于该值
        self.min_calc_offset = max(5.0 * scale, 2.5)

        self.frame_index = 0
        self.last_base_frame = None
        # 连续出现的0的个数
        self.consecutive_zero_frame_count = 0

        self.vehicle_token = None
        self.vehicle_info_cache = None
        self.is_vehicle_detector_triggering = False
        self.is_vehicle_existing = False
        # 车辆检测的间隔帧数
        self.vehicle_detect_frame_interval = 6
        # 上检测框的最后裁剪图
        self.last_below_vehicle_detect_view = None
        # 下检测框的最后裁剪图
        self.last_above_vehicle_detect_view = None
        # 检测车辆的最小轮廓大小
        self.min_detect_contour_size = min_detect_contour_size
        self.min_detect_valid_contour_size = 15
        self.min_detect_contour_sum_area_size = 400

        self.reverse_lane = reverse_lane
        self.detect_car_leave_by_video = detect_car_leave_by_video

        self.clear()

    def clear(self):
        """清理缓存
        """
        self.frames = []
        self.offsets = []
        self.speeds = deque(maxlen=7)
        self.found_vehicle = False
        self.first_frame = None
        self.last_base_frame = None
        self.frame_index = 0
        self.consecutive_zero_frame_count = 0
        self.video_saved_frame_count = 0
        self.video_writer = None
        self.vehicle_token = None
        self.vehicle_info_cache = None
        self.is_vehicle_existing = False
        self.last_below_vehicle_detect_view = None

    def vehicle_detector_trigger(self, token):
        self.is_vehicle_detector_triggering = True
        self.is_vehicle_existing = True

        if self.vehicle_token is None:
            self.vehicle_token = token
            self.vehicle_info_cache = vehicle_info_cache.VehicleInfoCache(
                token)
        elif self.vehicle_token != token:
            logging.warning('上一部车尚未离开前, 收到下一部车的地感触发信号, 忽略')

    def vehicle_detector_release(self):
        self.is_vehicle_detector_triggering = False

    def put(self, frame):
        """
        存入帧

        :return bool 车辆是否刚离开
        """
        if frame is None:
            return False

        if self.is_vehicle_existing and self.is_vehicle_detector_triggering is False:
            if len(self.frames) == 0:
                self.is_vehicle_existing = False
                return True

            if self.detect_car_leave_by_video is True:
                self.is_vehicle_existing = self._is_exist_vehicle(frame)
            else:
                self.is_vehicle_existing = False

            if self.is_vehicle_existing is False:
                logging.info('车辆离开: {}'.format(self.vehicle_token))
                self.vehicle_info_cache.close_video(frame)
                return True

        if self.is_vehicle_existing is False:
            return False

        frame = copy.deepcopy(frame)

        self.vehicle_info_cache.save_video_frame(frame)

        self.frame_index += 1

        if self.frame_index % self.vehicle_detect_frame_interval == 1:
            self.last_below_vehicle_detect_view = self.detect_below_vehicle_view_crop_op(
                frame)
            self.last_above_vehicle_detect_view = self.detect_above_vehicle_view_crop_op(
                frame)

        if self.frame_index == 1:
            self.frames.append(frame)
            self.first_frame = frame
            self.last_base_frame = frame
        else:
            # 与前一帧对比，计算出车辆位移
            offset_result = self._calc_offset(self.last_base_frame, frame)
            offset = offset_result.offset

            # 当取最后一帧的速度时, 如果self.speeds为空, 则返回None, 此时直接忽略
            if offset is None:
                return False

            if offset_result.could_append_to_last_speed:
                self.speeds.append(offset)

                # 小位移忽略不计，下一帧才能计算出更明显的位移变化 (如果是取的最后一帧的速度, 则不判断小位移逻辑)
                if offset != 0 and abs(offset) < self.min_calc_offset:
                    return False

            self.offsets.append(offset)
            self.last_speed_too_small = False

            # 丢掉位移为0的
            if offset != 0:
                self.frames.append(frame)
                self.last_base_frame = frame
                self.consecutive_zero_frame_count = 0
            else:
                # 为节省内存, 速度为0的帧不缓存图片
                self.frames.append(None)
                self.consecutive_zero_frame_count += 1
                # 连续多帧速度为0时, 更换基础帧, 防止因基础帧特征点过少或因帧相隔过远 而导致持续找不到特征点
                if self.consecutive_zero_frame_count == int(10 * self.frame_scale):
                    self.last_base_frame = frame
                    self.consecutive_zero_frame_count = 0

        return False

    def get(self):
        """获取拼接结果

        Returns:
            np.ndarray|None: 拼接图像，无图像返回None
        """
        img = self._splice()
        return img

    def splice_and_save_picture(self):
        img = self.get()
        elapsed_time = (datetime.datetime.now() -
                        self.vehicle_info_cache.trigger_time).total_seconds()
        logging.info('拼接完成, token:{}, 耗时: {}s, 帧数={}, 换算为帧率={}hz'.format(
            self.vehicle_token, elapsed_time, self.frame_index, round(self.frame_index / elapsed_time)))

        if img is None:
            logging.warning('拼接图片为空')
            return False

        if self.vehicle_info_cache is None:
            logging.error('保存图片时, vehicle_info_cache为None')
            return False

        self.vehicle_info_cache.save_body_image(img)
        return True

    def _is_exist_vehicle(self, frame):
        """
        通过图像判断是否有车
        """
        def find_contours(_frame):
            _contours = find_below_contours(_frame)
            _contours.extend(find_above_contours(_frame))
            return _contours

        def find_below_contours(_frame):
            detect_view = self.detect_below_vehicle_view_crop_op(_frame)

            _contours = _find_contours(
                self.last_below_vehicle_detect_view, detect_view)

            self.last_below_vehicle_detect_view = detect_view

            return _contours

        def find_above_contours(_frame):
            detect_view = self.detect_above_vehicle_view_crop_op(_frame)

            _contours = _find_contours(
                self.last_above_vehicle_detect_view, detect_view)

            self.last_above_vehicle_detect_view = detect_view

            return _contours

        def _find_contours(last_vehicle_detect_view, detect_view):
            thresh = cv2.absdiff(last_vehicle_detect_view, detect_view)
            thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(thresh, 10, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.erode(thresh, None, iterations=2)
            thresh = cv2.dilate(thresh, None, iterations=2)
            _contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return _contours

        def _is_exist_vehicle_contours(_contours):
            valid_sum_area = 0
            for c in _contours:
                contour_area = cv2.contourArea(c)

                if contour_area < self.min_detect_valid_contour_size:
                    continue

                if contour_area > self.min_detect_contour_size:
                    return True

                valid_sum_area += contour_area
                if valid_sum_area > self.min_detect_contour_sum_area_size:
                    return True

            return False

        if self.last_below_vehicle_detect_view is None:
            detect_vehicle_view = self.detect_below_vehicle_view_crop_op(frame)
            self.last_below_vehicle_detect_view = cv2.cvtColor(
                detect_vehicle_view, cv2.COLOR_BGR2GRAY)
            return self.is_vehicle_existing

        if self.frame_index % self.vehicle_detect_frame_interval != 1:
            return self.is_vehicle_existing

        contours = find_contours(frame)

        return _is_exist_vehicle_contours(contours)

    def _get_last_speed(self):
        """获取前几帧的速度
        """
        if len(self.speeds) == 0:
            return None

        return self._median_value(list(self.speeds))

    def _calc_offset(self, frame1, frame2):
        """计算车辆位移
        """

        def get_features(image):
            """
            提取特征点
            """
            orb = cv2.ORB_create()
            kp, des = orb.detectAndCompute(image, None)
            return kp, des

        def get_good_match(des1, des2):
            """对比特征点
            """
            bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)

            raw_matches = bf.match(des1, des2)
            _matches_mask = [1] * len(raw_matches)
            return raw_matches, _matches_mask

        def get_offset_range():
            """
            计算允许的位移范围
            """
            last_offset = self._get_last_speed()
            min_offset = 2 * self.frame_scale
            max_offset_when_last_is_zero = 6.5
            if last_offset is None:
                _positive_range = (min_offset, 100 * self.frame_scale)
                _negative_range = (-100 * self.frame_scale, -min_offset)
            elif last_offset == 0:
                _positive_range = (min_offset, max_offset_when_last_is_zero)
                _negative_range = (-max_offset_when_last_is_zero, -min_offset)
            elif last_offset > 0:
                _positive_range = (
                    max(0.2 * last_offset, min_offset), 5 * last_offset * (self.consecutive_zero_frame_count + 1))
                _negative_range = (-max_offset_when_last_is_zero, -min_offset)
            else:
                _positive_range = (min_offset, max_offset_when_last_is_zero)
                _negative_range = (
                    -5 * last_offset * (self.consecutive_zero_frame_count + 1), min(-0.2 * last_offset, -min_offset))
            return _negative_range, _positive_range

        def filter_invalid_match():
            result = []
            max_y_offset = 30 * self.frame_scale
            for x_offset, y_offset in x_y_offsets:
                if abs(y_offset) > max_y_offset:
                    continue

                if x_offset > 0:
                    if x_offset <= positive_range[1]:
                        result.append((x_offset, y_offset))
                elif x_offset < 0:
                    if negative_range[0] <= x_offset:
                        result.append((x_offset, y_offset))
            return result

        def filter_mini_offset_match():
            return [(x_offset, y_offset) for x_offset, y_offset in x_y_offsets if
                    (x_offset > 0 and positive_range[0] <= x_offset) or (
                x_offset < 0 and x_offset <= negative_range[1])]

        def get_negative_offsets():
            x_offsets = map(lambda x: x[0], x_y_offsets)
            _negative_offsets = filter(lambda x: x < 0, x_offsets)
            _negative_offsets = list(_negative_offsets)
            _negative_offset = self._median_value(
                _negative_offsets) if len(_negative_offsets) > 0 else 0
            return _negative_offset, _negative_offsets

        def get_positive_offsets():
            x_offsets = map(lambda x: x[0], x_y_offsets)
            _positive_offsets = filter(lambda x: x > 0, x_offsets)
            _positive_offsets = list(_positive_offsets)
            _positive_offset = self._median_value(
                _positive_offsets) if len(_positive_offsets) > 0 else 0
            return _positive_offset, _positive_offsets

        if log_offset_details:
            debug_print('frame_index', self.frame_index)
            debug_print('last_offset', self._get_last_speed())

        frame1 = self.calc_offset_view_crop_op(frame1)
        frame2 = self.calc_offset_view_crop_op(frame2)

        #frames = np.hstack([frame1, frame2])
        #cv2.imshow("pics", frames)
        # cv2.waitKey(0)

        kp_1, des_1 = get_features(frame1)
        kp_2, des_2 = get_features(frame2)

        # 若算不出速度，则取前几帧的中间速度
        if des_1 is None or des_2 is None or len(des_1) < 2 or len(des_2) < 2:
            debug_print('not feature points found')
            return OffSetResult(self._get_last_speed(), False)

        good_matches, matches_mask = get_good_match(des_2, des_1)

        img = cv2.drawMatches(frame2, kp_2, frame1, kp_1, good_matches, None, flags=2)
        # cv2.imshow("frame1", frame1)
        # cv2.imshow("frame2", frame2)
        # cv2.imshow("img", img)
        # cv2.waitKey()

        # 若算不出速度，则取前几帧的中间速度
        if len(good_matches) <= 2:
            debug_print('not match points found')
            return OffSetResult(self._get_last_speed(), False)

        # 计算允许的位移范围
        negative_range, positive_range = get_offset_range()

        if log_offset_details:
            debug_print('positive_range', positive_range)
            debug_print('negative_range', negative_range)
            debug_print('ori match list((x_offset, y_offset))',
                        [(round(kp_1[m.trainIdx].pt[0] - kp_2[m.queryIdx].pt[0], 2),
                          round(kp_1[m.trainIdx].pt[1] - kp_2[m.queryIdx].pt[1], 2)) for m in good_matches])

            self.draw_matches(frame1, frame2, kp_1, kp_2, good_matches, matches_mask,
                              file_name=str(self.frame_index) + '.jpg', vehicle_id=self.vehicle_token)

        x_y_offsets = [
            (kp_1[m.trainIdx].pt[0] - kp_2[m.queryIdx].pt[0], kp_1[m.trainIdx].pt[1] - kp_2[m.queryIdx].pt[1]) for m in
            good_matches]

        # 过滤乱跳的匹配点, 去掉 垂直距离过大的 和 移动距离过大的
        x_y_offsets = filter_invalid_match()

        # 过滤乱跳的匹配点后, 无匹配点，与前面找不到特征点的处理一致, 取前几帧的中间速度
        if len(x_y_offsets) == 0:
            debug_print('not valid match points found')
            return OffSetResult(self._get_last_speed(), False)

        # 过滤位移过小的匹配点
        x_y_offsets = filter_mini_offset_match()

        # 过滤位移过小的匹配点后无匹配点, 认为速度为0
        if len(x_y_offsets) == 0:
            debug_print('offset of match points is too small')
            return OffSetResult(0, True)

        # 计算正负位移
        positive_offset, positive_offsets = get_positive_offsets()
        negative_offset, negative_offsets = get_negative_offsets()

        if log_offset_details:
            debug_print('positive_offsets---', len(positive_offsets), round(positive_offset, 2),
                        [round(s, 2) for s in positive_offsets])
            debug_print('negative_offsets---',
                        len(negative_offsets), round(negative_offset, 2))

        # 当负速度的特征点数量 大于 正速度的特征点数量, 判断为倒车
        if len(negative_offsets) > len(positive_offsets):
            return OffSetResult(int(negative_offset), True)

        return OffSetResult(int(positive_offset), True)

    def _median_value(self, values):
        """取中值，数量为偶数时取较大一位
        """
        # return np.median(values)
        return sorted(values)[int(len(values) / 2)]

    def _mean_value(self, values):
        """取均值，数量为偶数时取较大一位
        """
        return np.mean(values)

    def _median_filter(self, values, radius=3):
        """中值滤波
        """
        if len(values) < radius:
            return values

        length = len(values)
        result = [None] * length

        # 切片
        for i in range(length):
            split = values[max(0, i - radius):min(i + radius + 1, length)]
            result[i] = self._median_value(split)

        return result

    def _mean_filter(self, values, radius=5):
        """均值滤波
        """
        if len(values) < radius:
            return values

        length = len(values)
        result = [None] * length

        # 切片
        for i in range(length):
            split = values[max(0, i - radius):min(i + radius + 1, length)]
            result[i] = self._mean_value(split)

        return result

    def draw_matches(self, img_1, img_2, kp_1, kp_2, matches, matches_mask, file_name=None, vehicle_id=None):
        """绘制特征点对比图像
        """
        os.makedirs('./imagesplicer_debug/{}'.format(vehicle_id),
                    exist_ok=True)
        if vehicle_id:
            file_path = os.path.join(
                './imagesplicer_debug', vehicle_id, file_name)
        else:
            file_path = os.path.join('./imagesplicer_debug', file_name)

        out_img = cv2.drawMatches(
            img_2, kp_2, img_1, kp_1, matches, None, matchesMask=matches_mask, flags=0)
        cv2.imwrite(file_path, out_img)

    def _offset_filter(self):
        """倒车过滤器
        倒车时位移与前一个位移相抵消，并删除被抵消的对应帧;
        丢弃位移为0的帧
        """
        debug_print('org', self.offsets)
        # 中值滤波重新计算位移
        self.offsets = self._median_filter(self.offsets)
        debug_print('alter', self.offsets, sum(self.offsets))

        # 删除尾部空白帧
        # while len(self.offsets) > 0 and self.offsets[-1] <= 0:
        #     del self.offsets[-1]

        new_offsets, new_frames = [], []

        for i, offset in enumerate(self.offsets):
            frame = self.frames[i]
            if offset == 0:
                continue
            elif offset > 0:
                new_offsets.append(offset)
                new_frames.append(frame)
            else:
                if len(new_offsets) > 0:
                    # 倒车，从缓存中减去位移
                    new_offsets[-1] += offset
                    new_frames[-1] = frame
                    # 若最后的值为负数，对数组进行压缩，丢掉速度为负的帧
                    while len(new_offsets) > 1 and new_offsets[-1] < 0:
                        new_offsets[-1] = new_offsets[-2] + new_offsets[-1]
                        del new_offsets[-2]
                        del new_frames[-2]
                    while len(new_offsets) > 0 and new_offsets[-1] < 0:
                        del new_offsets[-1]
                        del new_frames[-1]

        logging.info("frame count: frames:%d, flitered_frames:%d" %
                     (len(self.frames), len(new_frames)))

        self.offsets, self.frames = new_offsets, new_frames

        logging.info('frame count: %s, speed count: %s' %
                     (len(self.frames), len(self.offsets)))
        logging.info('calc speed: %s' % str(self.offsets))

    def _splice(self):
        """截取与拼接
        """
        # debug_print('self.offsets', self.offsets)
        # self._offset_filter()
        # self.offsets[0] = self.offsets[1]

        self._filter_zero_speed_frame()

        # 倒车
        if sum(self.offsets) < 0:
            return None

        # 帧数过少
        if len(self.frames) < self.min_splicer_frame_count:
            return None

        cut_frames = []
        for i, offset in enumerate(self.offsets):
            # 位移为0或倒车，不拼接
            if offset <= 0:
                continue

            # 根据位移长度截取每一帧
            frame = self.frames[i]

            # 第一帧，带车头
            if i == 0:
                cut_frame = Crop(
                    view=(0, self.height, 0, self.splice_x_pos + offset))(frame)
                cv2.imshow("crop", cut_frame)    
                cv2.waitKey()
            # 最后一帧，带车尾
            elif i == len(self.offsets) - 1:
                cut_frame = Crop(
                    view=(0, self.height, self.splice_x_pos, self.width))(frame)
            else:
                cut_frame = Crop(
                    view=(0, self.height, self.splice_x_pos, self.splice_x_pos + offset))(frame)
                # cv2.imshow("frame", frame)
                # cv2.imshow("crop", cut_frame)
                # cv2.waitKey()

            cut_frames.append(cut_frame)

        # 拼接画面
        if len(cut_frames) < self.min_splicer_frame_count:
            return None
        else:
            image = np.hstack(cut_frames)
            image = Flip(do_flip=self.reverse_lane)(image)
            cv2.imshow("image", image)
            cv2.waitKey()
            return image

    def _filter_zero_speed_frame(self):
        """清除速度为0的帧缓存和位移缓存"""
        new_frames = []
        new_offsets = []
        for i, frame in enumerate(self.frames):
            if i == 0:
                new_frames.append(frame)
                continue

            if frame is not None:
                new_frames.append(frame)
                new_offsets.append(self.offsets[i - 1])

        self.frames = new_frames
        self.offsets = new_offsets


def test(video_file):
    print('video_file:', video_file)

    import time
    start_time = time.time()

    scale = 0.5

    video = cv2.VideoCapture(video_file)
    # splicer = ImageSplicer(int(1280 * scale), int(720 * scale), (730 * scale, 1110 * scale, 240 * scale, 360 * scale),
    #                        (800 * scale, 1200 * scale, 60 * scale, 660 * scale), 340 * scale, scale)
    splicer = ImageSplicer(1280 * scale, 720 * scale, (800 * scale, 1240 * scale, 360 * scale, 540 * scale),
                           (600 * scale, 1040 * scale, 60 * scale, 660 * scale), 340 * scale, scale)

    veh_id = os.path.splitext(os.path.basename(video_file))[0]
    splicer.vehicle_detector_trigger(veh_id)

    frame_count = 0
    while True:
        ret, frame = video.read()
        if ret is False:
            break
         # cv2.imwrite("./imgs/{}.jpg".format(frame_count), frame)
        frame = Scale(scale=scale)(frame)

        assert splicer.put(frame) is False
        frame_count += 1

    splicer.vehicle_detector_release()

    # vehicle_leave = False
    # for i in range(10):
    #     if splicer.put(splicer.first_frame) is True:
    #         vehicle_leave = True
    #         break
    # assert vehicle_leave is True

    img = splicer.get()
    if img is not None:
        cv2.imwrite(
            './test_output/{}.jpg'.format(os.path.basename(video_file).split('.')[0]), img)
    else:
        print('{} splicer get None'.format(os.path.basename(video_file)))
    splicer.clear()

    print('{} cost time:{}, frame count:{}'.format(
        os.path.basename(video_file), time.time() - start_time, frame_count))


def test_dir_video(dir_path):
    import glob
    # for video_path in tqdm(sorted(glob.glob('{}/*.avi'.format(dir_path)))):
    #     test(video_path)
    video_path = 'F:/imagesplicer/test_video_dir/tooFast/44018B0D0420211019194637603.avi'
    test(video_file=video_path)

if __name__ == '__main__':
    if os.path.isdir('./test_output') is False:
        os.makedirs('./test_output')

    test_dir_video('test_video_dir/camera2_avi/error')
