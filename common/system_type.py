#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
this module use for identify the system type.
"""

import platform


def is_windows():
    return platform.system() == 'Windows'


def is_linux():
    return platform.system() == 'Linux'


def if_sys_type_is_supported():
    return is_windows() or is_linux()


def check_if_sys_type_is_supported():
    """
    check if current system type is supported, raise RuntimeError while not support.
    :raise RuntimeError: RuntimeError('current system type is not supported for this module')
    """
    if if_sys_type_is_supported() is False:
        raise RuntimeError('current system type is not supported for this module')


def test():
    if is_windows():
        print('Windows')
    elif is_linux():
        print('Linux')
    else:
        print('other system type')


if __name__ == '__main__':
    test()
