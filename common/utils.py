"""
Author       : xujiawei
Date         : 2021-03-15 16:02:08
Description  : 工具类
"""
import datetime
import os
import shutil
import logging
from collections import OrderedDict
from queue import Empty

from common.constant import *


class ImageFilePath(object):
    """生成图片文件路径
    """

    def __init__(self, file_prefix, save_pic_dir='./data/captured', dev_name='camera', backup_existed=False):
        # save_pic_dir = config['picture']['save_pic_dir']
        # dev_name = config['camera']['name']
        date = datetime.datetime.today().strftime('%Y-%m-%d')
        os.makedirs(os.path.join(save_pic_dir, date, dev_name), exist_ok=True)
        self.file_path = os.path.join(save_pic_dir, date, dev_name, file_prefix + '.jpg')

        if backup_existed and os.path.exists(self.file_path):
            bak_file_path = '{}.bak'.format(self.file_path)
            if os.path.isfile(bak_file_path):
                os.remove(bak_file_path)
            shutil.move(self.file_path, bak_file_path)

    def get(self):
        return self.file_path


class VideoFilePath(object):
    """生成视频文件路径
    """

    def __init__(self, file_prefix, save_dir='./data/video', dev_name='camera', backup_existed=False):
        date = datetime.datetime.today().strftime('%Y-%m-%d')
        os.makedirs(os.path.join(save_dir, date, dev_name), exist_ok=True)
        self.file_path = os.path.join(save_dir, date, dev_name, file_prefix + '.avi')

        if backup_existed and os.path.exists(self.file_path):
            bak_file_path = '{}.bak'.format(self.file_path)
            if os.path.isfile(bak_file_path):
                os.remove(bak_file_path)
            shutil.move(self.file_path, bak_file_path)

    def get(self):
        return self.file_path


class RecorderFile(object):
    """csv流水文件类

    Args:
        header (str): 文件头
    """

    def __init__(self, header, save_record_dir='./data/csv'):
        # save_record_dir = config['record']['save_record_dir']
        date = datetime.datetime.today().strftime('%Y-%m-%d')
        self.file_path = os.path.join(save_record_dir, date + '.csv')
        self.header = header
        self.exist = os.path.exists(self.file_path)
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def write(self, string):
        with open(self.file_path, 'a', encoding='utf8') as f:
            if not self.exist:
                f.write(self.header)
            f.write(string)
            f.flush()


class LimitedOrderedQueue(object):
    """限制长度且保持插入顺序的队列

    Args:
        max_len (int): 最大长度
    """

    def __init__(self, max_len=99):
        self.max_len = max_len
        self.queue = OrderedDict()

    def enqueue(self, key, value):
        while len(self.queue) >= self.max_len:
            self.queue.popitem(last=False)
        self.queue[key] = value

    def dequeue(self):
        if len(self.queue) > 0:
            return self.queue.popitem(last=False)[1]
        raise Empty()

    def update(self, key, value):
        if key in self.queue:
            self.queue[key] = value
        else:
            raise KeyError(key)

    def query(self, key):
        if key in self.queue:
            return self.queue[key]
        raise KeyError(key)

    def take(self, key):
        if key in self.queue:
            return self.queue.pop(key)
        raise KeyError(key)

    def has_key(self, key):
        return key in self.queue


class VehTypeCorrector:
    """车型修正逻辑
    """

    @staticmethod
    def adjust_veh_type_by_axis_count(veh_type, axis_count, prob):
        """根据轴数调整车型

        Args:
            veh_type (VehType): 当前车型
            axis_count (int): 轴数
            prob (float): 当前车型的置信度

        Returns:
            [VehType]: 调整后的车型
        """
        veh_type = VehType(veh_type)

        truck_list = [VehType.TRUCK_2, VehType.TRUCK_3,
                      VehType.TRUCK_4, VehType.TRUCK_5, VehType.TRUCK_6]
        special_list = [VehType.SPECIAL_2, VehType.SPECIAL_3,
                        VehType.SPECIAL_4, VehType.SPECIAL_5, VehType.SPECIAL_6]

        def is_truck(x):
            return x in truck_list

        def is_special_veh(x):
            return x in special_list

        def count_axis(x):
            """获取原车型轴数，客车、货1、专1不计入
            """
            if x in [VehType.TRUCK_2, VehType.SPECIAL_2]:
                return 2
            if x in [VehType.TRUCK_3, VehType.SPECIAL_3]:
                return 3
            if x in [VehType.TRUCK_4, VehType.SPECIAL_4]:
                return 4
            if x in [VehType.TRUCK_5, VehType.SPECIAL_5]:
                return 5
            if x in [VehType.TRUCK_6, VehType.SPECIAL_6]:
                return 6
            return 0

        def try_to_increase_veh_type(veh_type):
            """上调车型，货1、专1不计入
            """
            result = veh_type
            if is_truck(veh_type):
                result = truck_list[min(truck_list.index(veh_type) + 1, len(truck_list) - 1)]
            elif is_special_veh(veh_type):
                result = special_list[min(special_list.index(veh_type) + 1, len(special_list) - 1)]
            return result

        def try_to_decrease_veh_type(veh_type):
            """下调车型，货1、专1不计入
            """
            result = veh_type
            if is_truck(veh_type):
                result = truck_list[max(truck_list.index(veh_type) - 1, 0)]
            elif is_special_veh(veh_type):
                result = special_list[max(special_list.index(veh_type) - 1, 0)]
            return result

        result = veh_type
        org_axis_count = count_axis(veh_type)
        if axis_count == 0 or org_axis_count == 0 or axis_count == org_axis_count:
            result = veh_type
        # 允许上调车型
        elif axis_count == org_axis_count + 1:
            result = try_to_increase_veh_type(veh_type)
        # 置信度小于0.5的，才允许降车型
        elif axis_count == org_axis_count - 1 and prob < 0.5:
            result = try_to_decrease_veh_type(veh_type)

        if result != veh_type:
            logging.info('根据轴数调整车型结果：%s' % result.value)
        return result

    @staticmethod
    def adjust_veh_type_by_plate(veh_type, veh_plate_color, veh_plate):
        """根据车牌颜色、车牌内容调整车型

        Args:
            veh_type (VehType): 当前车型
            veh_plate_color (VehPlateColor): 外部传入车牌颜色
            veh_plate (str): 外部传入车牌信息

        Returns:
            [VehType]: 调整后的车型
        """

        veh_type = VehType(veh_type)
        veh_plate_color = VehPlateColor(veh_plate_color)

        result = veh_type

        # 0-蓝、4-渐变绿，客2转客1，货2转货1
        if veh_plate_color in [VehPlateColor.BLUE, VehPlateColor.GRADIENT_GREEN]:
            if VehType.COACH_2 == veh_type:
                result = VehType.COACH_1
            elif VehType.TRUCK_2 == veh_type:
                result = VehType.TRUCK_1
            elif VehType.SPECIAL_2 == veh_type:
                result = VehType.SPECIAL_1

        # 1-黄，5-黄绿，客1转客2，货1转货2，有学字的不转
        elif veh_plate_color in [VehPlateColor.YELLOW, VehPlateColor.YELLOW_GREEN]:
            if VehType.COACH_1 == veh_type and ('学' not in veh_plate):
                result = VehType.COACH_2
            elif VehType.TRUCK_1 == veh_type and ('学' not in veh_plate):
                result = VehType.TRUCK_2
            elif VehType.SPECIAL_1 == veh_type:
                result = VehType.SPECIAL_2

        if result != veh_type:
            logging.info('根据车牌调整车型结果：%s' % result.value)
        return result
