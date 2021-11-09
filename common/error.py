"""
Author       : xujiawei
Date         : 2021-03-12 10:55:29
Description  : 错误信息
FilePath     : /VehTypeCameraServer/trunk/common/error.py
"""
from enum import Enum
from collections import defaultdict


class ErrorCode(Enum):
    OK = '0'
    PARAM_MISSING = '1'
    CAMERA_OFFLINE = '2'
    PIPE_UNINIT = '3'
    PARAM_INVALID = '4'
    WAIT_FOR_RSP_TIMEOUT = '5'
    READ_VIEW_QUEUE_FAILED = '6'
    READ_EMPTY_FRAME_FROM_VIEW_QUEUE = '6'
    SYS_ERROR = '99'


_dict = defaultdict(lambda: 'unknown error')
_dict = {
    ErrorCode.OK: 'ok',
    ErrorCode.PARAM_MISSING: 'param missing',
    ErrorCode.CAMERA_OFFLINE: 'camera is offline',
    ErrorCode.PIPE_UNINIT: 'pipe uninitialized',
    ErrorCode.PARAM_INVALID: 'param invalid',
    ErrorCode.WAIT_FOR_RSP_TIMEOUT: '等待内部消息响应超时',
    ErrorCode.READ_VIEW_QUEUE_FAILED: '读取视频帧队列失败',
    ErrorCode.READ_EMPTY_FRAME_FROM_VIEW_QUEUE: '读取到空白的视频帧',
    ErrorCode.SYS_ERROR: 'system error'
}


class Error(Exception):
    def __init__(self, code: ErrorCode, extra_msg=None):
        msg = _dict[code]

        self.code = code
        self.msg = '%s: %s' % (msg, extra_msg) if extra_msg else msg
