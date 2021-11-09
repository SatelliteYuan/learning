"""
Author       : xujiawei
Date         : 2021-03-15 16:02:08
Description  : 常量
FilePath     : /VehTypeCameraServer/trunk/common/constant.py
"""
from enum import Enum


class VehType(Enum):
    """车型
    """
    UNKNOWN = '0'
    COACH_1 = '1'
    COACH_2 = '2'
    COACH_3 = '3'
    COACH_4 = '4'
    TRUCK_1 = '11'
    TRUCK_2 = '12'
    TRUCK_3 = '13'
    TRUCK_4 = '14'
    TRUCK_5 = '15'
    TRUCK_6 = '16'
    SPECIAL_1 = '21'
    SPECIAL_2 = '22'
    SPECIAL_3 = '23'
    SPECIAL_4 = '24'
    SPECIAL_5 = '25'
    SPECIAL_6 = '26'


class VehClass(Enum):
    """车种
    """
    UNKNOWN = 0
    COACH = 1
    TRUCK = 2


class VehPlateColor(Enum):
    """车牌颜色
    """
    BLUE = '00'
    YELLOW = '01'
    BLACK = '02'
    WHITE = '03'
    GRADIENT_GREEN = '04'  # 渐变绿
    YELLOW_GREEN = '05'
    BLUE_WHITE = '06'
    TEMP = '07',
    UNKNOWN = '09'
    GREEN = '11'
    RED = '12'


class TractorFlag(Enum):
    """拖车标记
    """
    UNKNOWN = '0'
    YES = '1'


class Events(Enum):
    """
    连接状态事件
    """
    GD_INFO = "地感信号"
    VEHICLE_PLATE_INFO = "车牌识别结果"
