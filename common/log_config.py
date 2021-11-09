"""
Author       : xujiawei
Date         : 2021-03-12 10:55:29
Description  : 日志配置
FilePath     : /VehTypeCameraServer/trunk/common/log_config.py
"""
import logging.config
import os

import yaml


def load_log_config():
    config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/log.yaml'))

    with open(config_file, 'r', encoding='utf8') as f:
        js = yaml.load(f, Loader=yaml.FullLoader)
        logging.config.dictConfig(js)
