"""
Author       : xujiawei
Date         : 2021-03-12 10:55:29
Description  : 图像处理类
FilePath     : /VehTypeCameraServer/trunk/common/image_operator.py
"""
import base64

import cv2
import numpy as np
import pickle


class EncoderDecoder(object):
    """编码器、解码器
    """
    CODEC_NUMPY = 0
    CODEC_BYTES = 1
    CODEC_STR = 2
    CODEC_BASE64_STR = 3
    CODEC_BASE64_BYTES = 4

    def __init__(self, src_codec, dst_codec):
        self.src_c, self.dst_c = src_codec, dst_codec

    def __call__(self, image):
        if self.src_c == self.CODEC_NUMPY and self.dst_c == self.CODEC_BASE64_BYTES:
            return self._np_2_base64_bytes(image)

        elif self.src_c == self.CODEC_NUMPY and self.dst_c == self.CODEC_BASE64_STR:
            return self._np_2_base64_str(image)

        elif self.src_c == self.CODEC_NUMPY and self.dst_c == self.CODEC_BYTES:
            return self._np_2_bytes(image)

        elif self.src_c == self.CODEC_NUMPY and self.dst_c == self.CODEC_STR:
            raise Exception('Can not encode numpy to str')
            # return self._np_2_str(image)

    def _np_2_base64_bytes(self, image):
        data = cv2.imencode('.jpg', image)[1]
        return base64.b64encode(data.tostring())

    def _np_2_bytes(self, image):
        data = cv2.imencode('.jpg', image)[1]
        return data.tostring()

    def _np_2_base64_str(self, image):
        data = cv2.imencode('.jpg', image)[1]
        return base64.b64encode(data.tostring()).decode('utf8')


class Rotate(object):
    """旋转
    """

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, image):
        (h, w) = image.shape[:2]
        (cx, cy) = (w / 2, h / 2)
        # 设置旋转矩阵
        M = cv2.getRotationMatrix2D((cx, cy), -self.angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # 计算图像旋转后的新边界
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy
        return cv2.warpAffine(image, M, (nW, nH))


class Scale(object):
    """
    size: (w,h)，优先级较高
    scale: 缩放比例
    """

    def __init__(self, size=None, scale=None):
        self.size = size
        self.scale = scale

    def __call__(self, image):
        if (self.size is None) and (self.scale is None):
            return image

        if (self.scale is not None) and int(self.scale) == 1:
            return image

        if self.size is not None:
            dst_size = self.size
        else:
            dst_size = (int(image.shape[1] * self.scale), int(image.shape[0] * self.scale))

        return cv2.resize(image, dst_size, interpolation=cv2.INTER_AREA)


class Flip(object):
    """翻转
    """

    def __init__(self, do_flip=False):
        self.do_flip = do_flip

    def __call__(self, image):
        return cv2.flip(image, 1) if self.do_flip else image


class Crop(object):
    """裁剪：top, bottom, left, right
    """

    def __init__(self, view=None, view_rate=None):
        self.view = view
        self.view_rate = view_rate

    def __call__(self, image):
        if self.view is None and self.view_rate is None:
            return image

        if self.view is not None:
            top, bottom, left, right = self.view
        elif self.view_rate is not None:
            height, width = image.shape[:2]
            top, bottom, left, right = (self.view_rate[0] * height, self.view_rate[1] * height,
                                        self.view_rate[2] * width, self.view_rate[3] * width)

        return image[int(top):int(bottom), int(left):int(right)]


class BrightnessCalculator(object):
    """计算y通道均值
    """

    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        mean_y = image[:, :, 0].mean()
        return round(mean_y)


class GammaCorrectCalculator(object):
    """计算gamma校正竖直
    """

    def _get_channels(self, image):
        # 2维的是灰度图，通道为1
        # 3维的是彩色图，通道为3
        return 1 if image.ndim == 2 else 3

    def _calc_gamma(self, channel):
        return - 0.3 / np.log10(channel / 256)

    def __call__(self, image):
        channels = self._get_channels(image)
        # 处理灰度图
        if channels == 1:
            avg_g = np.mean(image)
            gamma = self._calc_gamma(avg_g)
            return gamma

        # 处理彩色图
        elif channels == 3:
            b, g, r = cv2.split(image)
            avg_b, avg_g, avg_r = b.mean(), g.mean(), r.mean()
            gamma_b, gamma_g, gamma_r = self._calc_gamma(avg_b), self._calc_gamma(avg_g), self._calc_gamma(avg_r)
            gamma = np.mean([gamma_b, gamma_g, gamma_r])
            return gamma


class GlobalGammaCorrect(object):
    """全局gamma校正
    """

    def _calc_table(self, gamma):
        # 建立映射表
        table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        # 颜色值为整数
        table = np.round(np.array(table)).astype(np.uint8)
        return table

    def __call__(self, image):
        gamma = GammaCorrectCalculator()(image)
        table = self._calc_table(gamma)
        return cv2.LUT(image, table)


class LocalGammaCorrect(object):
    """局部gamma校正，自适应调整
    """

    def __call__(self, image):
        b, g, r = cv2.split(image)
        mean_t = 255 - (b + g + r) / 3
        mask = cv2.GaussianBlur(mean_t, (41, 41), cv2.CV_8U)
        table = np.power(2, (128 - mask) / 128)
        image = 255 * np.power(image / 255, np.expand_dims(table, axis=2))
        return image


def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


def load_variable(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


def perspective(image, src_matrix_str, dst_matrix_str):
    """
    透视变换
    src_matrix: 原始图上任意四点[(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    dst_matrix: 调整后的四点位置[(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    """
    w = image.shape[1]
    h = image.shape[0]
    src_matrix = eval(src_matrix_str)
    dst_matrix = eval(dst_matrix_str)

    matrix = cv2.getPerspectiveTransform(np.float32(src_matrix), np.float32(dst_matrix))
    image = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))
    return image


def calibrate(image, matrix, dist, crop=False):
    """
    畸变校正

    image: 图像
    matrix: 内参矩阵
    dist: 透镜畸变系数
    crop: 是否裁剪，若裁剪，返回的图像宽度高度与原始图像不同

    return: 转换后的图像，宽带，高度
    """
    # image = cv2.resize(image, (480, 640), interpolation=cv2.INTER_AREA)
    h0, w0 = image.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(matrix, dist, (w0, h0), 1, (w0, h0))
    # print(roi)

    x, y, w, h = roi
    image = cv2.undistort(image, matrix, dist, None, new_camera_mtx)

    if not crop:
        return image, w0, h0
    else:
        image = image[y:y + h, x:x + w]
        return image, w, h


def calibrate_img_example():
    mtx, dist = load_variable('calibration_170.pickle')
    img = cv2.imread('image_file_to_calibrate')
    img, w, h = calibrate(img, mtx, dist, crop=True)
    print(w, h)
    cv2.imwrite('./test.jpg', img)


def calibrate_video_example():
    input_video = 'path_to_video_file'
    output_video = 'path_to_output_video_file'

    mtx, dist = load_variable('calibration.pickle')

    input_stream = cv2.VideoCapture(input_video)
    output_stream = None

    while True:
        ok, frame = input_stream.read()
        if not ok:
            break

        frame, w, h = calibrate(frame, mtx, dist, crop=True)
        if not output_stream:
            output_stream = cv2.VideoWriter(output_video,
                                            int(input_stream.get(cv2.CAP_PROP_FOURCC)),
                                            input_stream.get(cv2.CAP_PROP_FPS),
                                            (w, h), True)
        output_stream.write(frame)

    input_stream.release()
    if output_stream:
        output_stream.release()


class Grid(object):
    def __init__(self, row=1, column=1):
        assert 0 < row < 10
        assert 0 < column < 10
        self.row, self.column = row, column

    def __call__(self, image):
        h, w = image.shape[:2]
        for x in range(0, w, int(w / self.column)):
            cv2.line(image, (int(x), 0), (int(x), h), (0, 0, 255), 1)
        for y in range(0, h, int(h / self.row)):
            cv2.line(image, (0, int(y)), (w, int(y)), (0, 0, 255), 1)

        return image
