#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# cython: language_level=3

import os
from common import system_type


def increase_priority(step_priority=1):
    try:
        if system_type.is_linux():
            # 根据 pid 获取优先级
            which = os.PRIO_PROCESS
            who = os.getpid()
            pro = os.getpriority(which, who)
            # 调高进程优先级(priority 是范围在 -20 至 19 的值。 默认优先级为 0；较小的优先级数值会更优先被调度)
            os.setpriority(which, who, pro - step_priority)
            return True
    except Exception:
        return False

