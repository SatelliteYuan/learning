#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# cython: language_level=3

import datetime
import logging
import os
import shutil

import cv2

from common.constant import VehPlateColor, VehType
from common import utils

dev_name = 'camera'
save_pic_dir = './data/captured'
save_video = False
video_save_dir = './data/video'
save_record_dir = './data/csv'


def load_config():
    from common.config import config
    global dev_name, save_pic_dir, save_video, video_save_dir, save_record_dir
    dev_name = config['camera']['name']
    save_pic_dir = config['picture']['save_pic_dir']
    save_video = config['video']['save_video']
    video_save_dir = config['video']['save_video_dir']
    save_record_dir = config['record']['save_record_dir']


class VehicleInfoCache:
    """
    车辆元素缓存类
    负责缓存各种车型信息到文件
    """

    def __init__(self, token):
        self.token = token
        self.got_plate_info_from_robot = False
        self.color = VehPlateColor.UNKNOWN.value
        self.plate = ''
        self.type = VehType.UNKNOWN.value
        self.type_by_recognize = VehType.UNKNOWN.value
        self.trigger_time = datetime.datetime.now()
        self.axis_count = 0
        self.type_probability = 0
        self.is_trailer = False  # 是否为拖车
        self.is_be_towed = False  # 是否为被拖车
        self.splited_vehicle_token = None  # 分割出的车辆id
        self.veh_class = 0
        self.veh_class_probability = 0
        self.cockpit_height = 0
        self.cockpit_height_probability = 0

        self.video_saved_frame_count = 0
        self.max_video_frame_count = 1800
        self.video_writer = None

    def to_json_dict(self):
        _dict = self.__dict__
        _dict['trigger_time'] = self.trigger_time.strftime('%Y-%m-%dT%H:%M:%S')
        return _dict

    def from_json_dict(self, json_dict):
        json_dict['trigger_time'] = datetime.datetime.strptime(json_dict['trigger_time'], '%Y-%m-%dT%H:%M:%S')
        self.__dict__ = json_dict

    def set_plate_info(self, plate, color):
        """绑定车牌信息
        """
        self.color = color
        self.plate = plate
        self.got_plate_info_from_robot = True

    def set_type_by_recognize(self, type_by_recognize, probability):
        """绑定识别车型
        """
        self.type_by_recognize = str(type_by_recognize)
        self.type_probability = probability

    def set_veh_class(self, veh_class, veh_class_probability):
        self.veh_class = veh_class
        self.veh_class_probability = veh_class_probability

    def set_cockpit_height(self, cockpit_height, cockpit_height_probability):
        self.cockpit_height = cockpit_height
        self.cockpit_height_probability = cockpit_height_probability

    def set_axis_count(self, axis_count):
        self.axis_count = axis_count

    def adjust_veh_type(self):
        if self.got_plate_info_from_robot:
            self.type = utils.VehTypeCorrector.adjust_veh_type_by_plate(self.type_by_recognize, self.color,
                                                                        self.plate).value
        else:
            self.type = self.type_by_recognize

        self.type = utils.VehTypeCorrector.adjust_veh_type_by_axis_count(self.type, self.axis_count,
                                                                         self.type_probability).value

    def distinguish_trailer(self):
        """
        分辨拖车
        """
        if self.type in (VehType.SPECIAL_1.value, VehType.SPECIAL_2.value) \
                and self.splited_vehicle_token is not None:
            self.is_trailer = True
            self.veh_class = 12
        elif self.veh_class == 12:
            self.is_trailer = True

    def restore_trailer(self):
        """
        还原拖车与被拖车为一部车, 当配置为将拖车当一部车处理时使用
        """
        logging.info('将拖车还原为1部车')
        self.type = VehType.SPECIAL_3.value
        body_file_path = utils.ImageFilePath(self.token).get()
        bak_file_path = body_file_path + '.bak'
        if os.path.isfile(bak_file_path):
            if os.path.isfile(body_file_path):
                os.remove(body_file_path)
            shutil.move(bak_file_path, body_file_path)
        self.splited_vehicle_token = None
        self.is_trailer = False

    def save_record_file(self):
        """token，车型，车牌颜色，车牌，模型识别车型
        """
        header = 'vehToken,vehType,vehPlateColor,vehPlate,' \
                 'vehTypeByRecognize,probability,axisCount,vehClass,cockpitHeight,time\n'
        string = '%s,%s,%s,%s,%s,%s,%d,%d,%d,%s' % (
            self.token, self.type, self.color, self.plate,
            self.type_by_recognize, self.type_probability, self.axis_count, self.veh_class, self.cockpit_height,
            datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        )
        utils.RecorderFile(header, save_record_dir).write(string + '\n')
        logging.info('保存识别流水')
        logging.info(header.strip() + ':' + string)

    def save_body_image(self, body_image):
        image_file_path = utils.ImageFilePath(self.token, save_pic_dir, dev_name, backup_existed=True).get()
        cv2.imwrite(image_file_path, body_image)
        logging.info('save vehicle image to %s' % image_file_path)

    def save_video_frame(self, frame):
        try:
            if save_video is False:
                return

            if self.video_writer is None:
                file_path = utils.VideoFilePath(self.token, video_save_dir, dev_name, backup_existed=True).get()
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                h, w = frame.shape[:2]
                self.video_writer = cv2.VideoWriter(file_path,
                                                    cv2.VideoWriter_fourcc(*'XVID'),
                                                    60,
                                                    (w, h), True)

            if self.video_saved_frame_count > self.max_video_frame_count:
                return

            self.video_writer.write(frame)
            self.video_saved_frame_count += 1

        except Exception as e:
            logging.error('Exception at save_video_frame:' + str(e))

    def close_video(self, frame):
        try:
            if save_video is False:
                return

            if self.video_writer is None:
                return

            self.video_writer.write(frame)
            self.video_writer.release()
            self.video_saved_frame_count = 0
            self.video_writer = None
            file_path = utils.VideoFilePath(self.token, video_save_dir, dev_name, backup_existed=False).get()
            logging.info('save vehicle video to %s' % file_path)

        except Exception as e:
            logging.error('Exception at close_video:' + str(e))

    def get_body_image(self):
        file_path = utils.ImageFilePath(self.token).get()

        if os.path.isfile(file_path) is False:
            return None

        return cv2.imread(file_path)

    def get_body_image_file_path(self):
        file_path = utils.ImageFilePath(self.token).get()

        if os.path.isfile(file_path) is False:
            return None

        return file_path

    def get_video_file_path(self):
        file_path = utils.VideoFilePath(self.token, video_save_dir, dev_name).get()

        if os.path.isfile(file_path) is False:
            return None

        return file_path
