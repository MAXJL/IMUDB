
import torch
import torch.nn.functional as F
import pywt
import numpy as np



def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo, dec_hi], dim=0)  # Shape: [2, filter_size]
    dec_filters = dec_filters.repeat(in_size, 1, 1)  # Shape: [in_size*2, filter_size]
    dec_filters = dec_filters.view(in_size * 2, 1, -1)  # Shape: [in_size*2, 1, filter_size]
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type)
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type)
    rec_filters = torch.stack([rec_lo, rec_hi], dim=0)
    rec_filters = rec_filters.repeat(out_size, 1, 1)  # Shape: [out_size*2, filter_size]
    rec_filters = rec_filters.view(out_size * 2, 1, -1)  # Shape: [out_size*2, 1, filter_size]
    # print(f"Decomposition filter shape: {dec_filters.shape}")
    # print(f"Reconstruction filter shape: {rec_filters.shape}")
    return dec_filters, rec_filters


def wavelet_transform_1d(x, filters):
    device = x.device
    filters = filters.to(device)
    b, c, w = x.shape
    pad = (filters.shape[2] // 2 - 1)
    x = F.conv1d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 2, w // 2)  # 进行1D卷积后的形状
    return x


def inverse_wavelet_transform_1d(x, filters):
    device = x.device
    filters = filters.to(device)
    b, c, _, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1)

    x = x.reshape(b, c * 2, w_half)
    filters = filters.repeat(c // (filters.shape[0] // 2), 1, 1)
    x = F.conv_transpose1d(x, filters, stride=2, groups=c, padding=pad)

    # 重新排列时间步，每两个元素交换位置
    # 创建一个索引映射
    indices = torch.arange(x.shape[-1])
    # 适用于偶数长度的序列
    indices = indices.view(-1, 2).flip(dims=[1]).reshape(-1)
    x = x.index_select(2, indices.to(device))

    return x





if __name__ == '__main__':
     # Define wavelet type and level.

    # data = torch.tensor([[1, 2, 3, 4, 5, 6]])
    data = torch.rand(1, 6, 8)


    wavelet_type = 'db1'
    in_channels = out_channels = 6
    levels = 2
    # Create filters
    dec_filters, rec_filters = create_wavelet_filter(wavelet_type, in_channels, out_channels)
    wt_output = wavelet_transform_1d(data, dec_filters)

    # 第二级变换：低频部分
    low_freq_part = wt_output[:, :, 0, :]
    wt_output_level2_low = wavelet_transform_1d(low_freq_part, dec_filters)

    # 第二级变换：高频部分
    high_freq_part = wt_output[:, :, 1, :]
    wt_output_level2_high = wavelet_transform_1d(high_freq_part, dec_filters)

    # 拼接所有二级变换的结果
    concatenated_features = torch.cat([
        wt_output_level2_low[:, :, 0, :],    # 第一级低频后的第二级低频
        wt_output_level2_low[:, :, 1, :],    # 第一级低频后的第二级高频
        wt_output_level2_high[:, :, 0, :],   # 第一级高频后的第二级低频
        wt_output_level2_high[:, :, 1, :]    # 第一级高频后的第二级高频
    ], dim=1)  # 按特征维度拼接

    print("拼接后的特征维度:", concatenated_features.shape)

        # 分离优化后的特征
    second_level_low_low = concatenated_features[:, 0:6, :]  # 每个部分的通道数仍为6
    second_level_low_high = concatenated_features[:, 6:12, :]
    second_level_high_low = concatenated_features[:, 12:18, :]
    second_level_high_high = concatenated_features[:, 18:24, :]

    # 重组第二级变换结果以进行逆变换
    second_level_low_reconstructed = torch.cat([second_level_low_low.unsqueeze(2), second_level_low_high.unsqueeze(2)], dim=2)
    second_level_high_reconstructed = torch.cat([second_level_high_low.unsqueeze(2), second_level_high_high.unsqueeze(2)], dim=2)

    # 第二级逆小波变换
    first_level_low_reconstructed = inverse_wavelet_transform_1d(second_level_low_reconstructed, rec_filters)
    # print("第一次变换的低频部分",low_freq_part )
    # print("第一级低频重构后:", first_level_low_reconstructed)

    first_level_high_reconstructed = inverse_wavelet_transform_1d(second_level_high_reconstructed, rec_filters)

    # 重组第一级变换结果以进行最终逆变换
    first_level_reconstructed = torch.cat([first_level_low_reconstructed.unsqueeze(2), first_level_high_reconstructed.unsqueeze(2)], dim=2)

    # 第一级逆小波变换，重构原始信号
    original_data_reconstructed = inverse_wavelet_transform_1d(first_level_reconstructed, rec_filters)

    print("原始数据的维度:", data.shape)
    print("原始数据:")
    print(data)
    print("重构后的数据维度:")
    print(original_data_reconstructed.shape)
    # 输出重构的原始数据
    print("重构后的原始数据:")
    print(original_data_reconstructed)
    atol = 1e-6  # 绝对误差阈值
    rtol = 1e-5  # 相对误差阈值
    print("重构是否与原信号相同:", np.allclose(data, original_data_reconstructed, atol=atol, rtol=rtol))







