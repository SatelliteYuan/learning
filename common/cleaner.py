'''
Author: xujiawei
Date: 2021-03-19 10:30:37
Description: 清理过期文件
'''
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from threading import Thread

import logging


class Cleaner(object):
    ''' 定时清理 '''

    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai',
    )

    monitor_dir = {}

    def __init__(self):
        self.start_cleaner_thread()

    def add_monitor(self, path, backup_days):
        self.monitor_dir[path] = backup_days

    def get_now(self):
        # 协调世界时
        utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
        # 北京时间
        beijing_now = utc_now.astimezone(self.SHA_TZ)
        return beijing_now

    def start_cleaner_thread(self):
        def run():
            while True:
                today = self.get_now().date()

                for dir, keep_days in self.monitor_dir.items():
                    for item in os.listdir(dir):
                        try:
                            day = datetime.strptime(item[:10], '%Y-%m-%d').date()
                            out_of_day = today - timedelta(days=keep_days)
                            if day < out_of_day:
                                del_item = os.path.join(dir, item)
                                if os.path.isdir(del_item):
                                    shutil.rmtree(del_item)
                                elif os.path.isfile(del_item):
                                    os.remove(del_item)
                                logging.info('remove out-of-day item: %s' % del_item)
                        except Exception as e:
                            pass

                time.sleep(60 * 60)

        Thread(target=run, daemon=True).start()


cleaner = Cleaner()

# if __name__ == '__main__':
#     Cleaner().start_cleaner_thread()
#     while True:
#         print('running')
#         time.sleep(30)
