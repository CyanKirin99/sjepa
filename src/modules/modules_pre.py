# Copyright (c) Chenye Su, Wuhan University.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn
import numpy as np
from scipy.signal import savgol_filter


class FilterResampler(nn.Module):
    def __init__(self, window_size=11, poly_order=2, resampling_rate=1, spec_len=2151, total_len=2160):
        # 初始化函数，设置滤波窗口大小、多项式阶数、重采样率和序列长度
        super(FilterResampler, self).__init__()
        self.resampling_rate = resampling_rate
        self.window_size = window_size
        self.poly_order = poly_order
        self.spec_len = spec_len
        self.total_len = total_len
        self.front_endpoint = 350
        self.back_endpoint = 2500

    def forward(self, x):
        """
        对输入的批量数据进行滤波和重采样
        :param x ，尺寸为(batch_size, spec_len)
        :return: x_tensor (torch tensor)，形状为(batch_size, spec_len+pad_len)
        """

        x_resampled = np.zeros((len(x), self.spec_len)).astype(np.float32)
        for i, x_ in enumerate(x):
            x_filtered = self.sg_filter(x_[~np.isnan(x_)])
            x_resampled[i] = self.resample(x_filtered, self.resampling_rate)
        # 填充到指定长度total_len
        x_padded = np.pad(x_resampled, ((0, 0), (0, self.total_len - self.spec_len)), mode='constant',
                          constant_values=0)
        return x_padded

    @staticmethod
    def sg_filter(x):
        """
        对输入的数据进行Savitzky-Golay滤波
        :param x: (numpy array)，尺寸为(spec_len + 3,)
        :return: y (numpy array)，形状为(spec_len + 3,)
        """
        start, end, spec_len, refl = x[0], x[1], x[2], x[3:]

        sampling_rate = (end - start + 1) / spec_len
        if sampling_rate <= 1.1:
            window_size, poly_order = 21, 2
        elif sampling_rate >= 9.9:
            window_size, poly_order = 5, 2
        else:
            window_size, poly_order = 11, 2
        refl = savgol_filter(refl, window_size, poly_order)

        y = np.concatenate([x[:3], refl]).astype(np.float32)
        return y

    def resample(self, x, resampling_rate):
        """
        对滤波后的数据进行重采样
        :param x: (numpy array)，尺寸为(spec_len+3,)
        :param resampling_rate: int，采样分辨率
        :return: y (numpy array)，尺寸为(spec_len,)
        """
        start, end, spec_len, refl = x[0], x[1], x[2], x[3:]

        original_wavelengths = np.linspace(start, end, refl.shape[0])
        target_wavelengths = np.linspace(start, end, int((end - start) / resampling_rate) + 1)
        refl_resampled = np.interp(target_wavelengths, original_wavelengths, refl).astype(np.float32)

        if len(target_wavelengths) == self.spec_len:
            y = refl_resampled
        else:
            padding_front = np.zeros(int(target_wavelengths[0]) - self.front_endpoint, dtype=np.float32)
            padding_back = np.zeros(self.back_endpoint - int(target_wavelengths[-1]), dtype=np.float32)
            y = np.concatenate([padding_front, refl_resampled, padding_back])
        return y
