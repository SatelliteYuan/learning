import ctypes
import multiprocessing
import multiprocessing.sharedctypes
import logging

import numpy as np


class VideoFrameQueue:
    def __init__(self, image_width, image_height, image_channel, queue_max_length=600):
        self.notify_queue = multiprocessing.Queue()

        # 图片数组
        self.image_shape = (image_height, image_width, image_channel)
        self.image_queue_shape = (queue_max_length, self.image_shape[0], self.image_shape[1], self.image_shape[2])
        self.image_queue_byte_size = \
            self.image_queue_shape[0] * self.image_queue_shape[1] * \
            self.image_queue_shape[2] * self.image_queue_shape[3]
        self.shared_memory = multiprocessing.sharedctypes.RawArray(ctypes.c_uint8, self.image_queue_byte_size)
        self.shm_image_array = None

        # 读写索引数组, 0为读索引, 1为写索引
        self.shm_rw_positions = multiprocessing.sharedctypes.RawArray(ctypes.c_uint64, [0, 0])

        self.recording = False
        self.is_warning = False

    def close(self):
        """
        退出时, 应调用close(), 以使阻塞中的read()能正常退出.
        :return:
        """
        if self.notify_queue:
            self.notify_queue.put('closed')

    def empty(self):
        return self.shm_rw_positions[0] == self.shm_rw_positions[1]

    read_index_equal_write_index = empty

    def read(self):
        """
        读取一帧图片或消息
        :return: is_success, image, msg. 若is_success为False, 表示已经停止使用该视频帧共享内存.
        """
        msg = self.notify_queue.get()
        if msg == 'closed':
            return False, None, msg

        elif msg == 'ok':
            image = self._pop()

            if image is not None:
                return True, image, msg
            else:
                # 若image为空, 表示队列曾经溢出, 有消息对应的图片已经被覆盖, 消息的数量大于队列中的图片数引起.
                # 此时忽略溢出帧, 等待下一帧的消息.
                return self.read()

        else:
            return True, None, msg

    def _pop(self):
        if self.empty():
            return None

        self.shm_image_array = np.ndarray(self.image_queue_shape, dtype=np.uint8, buffer=self.shared_memory)
        image = self.shm_image_array[self.shm_rw_positions[0]]

        self.increase_read_index()

        return image

    def increase_read_index(self):
        next_read_position = self.shm_rw_positions[0] + 1
        if next_read_position == self.image_queue_shape[0]:
            next_read_position = 0
        self.shm_rw_positions[0] = next_read_position

    def write(self, image):
        if image.shape != self.image_shape:
            raise RuntimeError('shape of image is not same with image queue')

        next_write_position = self.shm_rw_positions[1] + 1
        if next_write_position == self.image_queue_shape[0]:
            next_write_position = 0

        if next_write_position == self.shm_rw_positions[0]:
            if self.is_warning is False:
                logging.warning('图片队列缓存已满')
                self.is_warning = True
            # 缓存已满, 直接返回
            return

        if self.is_warning:
            logging.info('图片队列缓存已恢复')
            self.is_warning = False

        self.shm_image_array = np.ndarray(self.image_queue_shape, dtype=np.uint8, buffer=self.shared_memory)
        self.shm_image_array[self.shm_rw_positions[1]] = image

        self.shm_rw_positions[1] = next_write_position

        if self.notify_queue:
            self.notify_queue.put('ok')

    def push_msg(self, msg):
        if self.notify_queue:
            self.notify_queue.put(msg)


def test_write(queue, count):
    import time
    time.sleep(0.5)
    for i in range(count):
        begin_time = time.time()

        image = np.zeros([400, 400, 3], dtype=np.uint8)
        image[i % 400][0][0] = 255

        queue.write(image)

        print('write:{}, cost:{}'.format(i, time.time() - begin_time))

    queue.close()


def test_read(queue):
    import time
    i = 0
    while True:
        begin_time = time.time()

        ret, image = queue.read()
        if ret is False:
            break

        print('read:{}, cost:{}'.format(i, time.time() - begin_time))
        i += 1

    queue.close()


def test():
    image_queue = VideoFrameQueue(image_width=400, image_height=400, image_channel=3, queue_max_length=200)

    writer = multiprocessing.Process(target=test_write, args=(image_queue, 700))
    reader = multiprocessing.Process(target=test_read, args=(image_queue,))

    writer.start()
    reader.start()
    writer.join()
    reader.join()


if __name__ == '__main__':
    test()
