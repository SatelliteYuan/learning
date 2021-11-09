'''
Author       : xujiawei
Date         : 2021-08-13 11:42:31
Description  : description
'''
import os

import yaml

# 默认配置，启动时输出到 default_config.yaml
default_config = {
    'server': {'host': '0.0.0.0', 'port': 9965},
    'picture': {'backup_days': 2, 'save_pic_dir': './data/captured', 'delay_before_capture_sec': 0.5, 'capture_count': 3, 'capture_interval_sec': 0.5},
    'video': {'save_video': False, 'backup_days': 2, 'save_video_dir': './data/video'},
    'record': {'backup_days': 180, 'save_record_dir': './data/csv'},
    'vehicle_type_recognization': {'url': 'http://0.0.0.0:7000/prediction/vehicle_type_recognization'},
    'vehicle_detection': {'enable': True, 'url': 'http://0.0.0.0:7000/prediction/vehicle_detection', 'min_vehicle_bottom_line': 0.75},
    'vehicle_class_classification': {'enable': True, 'url': 'http://0.0.0.0:7000/prediction/vehicle_class_classification'},
    'cockpit_height_classification': {'enable': True, 'url': 'http://0.0.0.0:7000/prediction/cockpit_height_classification'},
    'vehicle_type_agent_server': {'enable': True, 'url': 'http://127.0.0.1:9201', 'send_video': False},
    'lane': {'binding_gd_index': 3, 'reverse_lane': False, 'wait_for_vehplate_time_sec': 0, 'gd_delay_time_sec': 0},
    'camera': {'name': 'imx290_170', 'type': 'imx290', 'device': 0, 'rotate': 90, 'scale': 0.5, 'brightness_range': '(90, 150)', 'exposure_range': '(1, 80)', 'gain_range': '(0, 100)', 'auto_exposure_hour_range': '(0,0)', 'default_exposure': 10},
    'splicer': {'frame_width': 360, 'frame_height': 640, 'detect_vehicle_view': '(400, 620, 170, 260)', 'calc_offset_view': '(400, 620, 30, 330)', 'splice_x_pos': 170, 'min_detect_contour_size': 150, "detect_car_leave_by_video": False},
    'ui_server': {'url': 'http://127.0.0.1:5000'},
    'processes_message_queue_type': 'RedisMessageQueue'
}

config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/default_config.yaml'))
special_config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/special_config.yaml'))


def load(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf8') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    else:
        cfg = {}
    return cfg


def export(config, file_path):
    with open(file_path, 'w', encoding='utf8') as w:
        yaml.dump(config, w, default_flow_style=False)


def combine(older, newer):

    for k, v in newer.items():
        if not isinstance(v, dict):
            older[k] = newer[k]
        else:
            combine(older[k], newer[k])

    return older


export(default_config, config_file)
special_config = load(special_config_file)
config = combine(default_config, special_config)
