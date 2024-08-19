import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import pywt
import pywt.data
import torch


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
    
    return dec_filters, rec_filters

def wavelet_transform_1d(x, filters):
    b, c, w = x.shape
    pad = (filters.shape[2] // 2 - 1)
    x = F.conv1d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 2, w // 2)  # 进行1D卷积后的形状
    return x

def inverse_wavelet_transform_1d(x, filters):
    b, c, _, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1)
    x = x.reshape(b, c * 2, w_half)
    x = F.conv_transpose1d(x, filters, stride=2, groups=c, padding=pad)
    return x



class MultiLayerWTConv1d(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, kernel_size=5, wt_levels=2):
        super(MultiLayerWTConv1d, self).__init__()
        
        # 确保输入输出维度一致
        assert in_channels == out_channels
        
        # 创建多个 WTConv1d 层
        self.layers = nn.ModuleList([
            WTConv1d(in_channels, out_channels, kernel_size, wt_levels=wt_levels)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        # 依次通过每一层 WTConv1d
        for layer in self.layers:
            x = layer(x)
        return x




class WTConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=2, wt_type='db1'):
        super(WTConv1d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform_1d, filters = self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform_1d, filters = self.iwt_filter)

        self.base_conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1,in_channels,1])

        wavelet_out_channels = in_channels * 2  
        self.wavelet_convs = nn.ModuleList(
            [nn.Conv1d(wavelet_out_channels, wavelet_out_channels, kernel_size, padding='same', stride=1, dilation=1, groups=wavelet_out_channels, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, wavelet_out_channels, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv1d(x_in, self.stride_filter, bias=None, stride=self.stride, groups=in_channels)
        else:
            self.do_stride = None


 

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) :
                curr_pads = (0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:,:,0]
            
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 2, shape_x[3])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:,:,0])
            x_h_in_levels.append(curr_x_tag[:,:,1])

        next_x_ll = 0

        for i in range(self.wt_levels-1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h.unsqueeze(2)], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0
        
        x = self.base_scale(self.base_conv(x))
        x = x + x_tag
        
        if self.do_stride is not None:
            x = self.do_stride(x)

        return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.weight = nn.Parameter(torch.ones(dims) * init_scale)
        if init_bias != 0:
            self.bias = nn.Parameter(torch.ones(dims) * init_bias)
        else:
            self.bias = None
    
    def forward(self, x):
        weight = self.weight
        if weight.shape[1] != x.shape[1]:
            weight = weight.expand_as(x)
        if self.bias is not None:
            return torch.mul(weight, x) + self.bias
        else:
            return torch.mul(weight, x)



if __name__ == '__main__':
    # 定义输入的参数
    batch_size = 1024  # Batch size
    sequence_length = 30  # Sequence length (S)
    feature_dim = 6  # Feature dimension (D)

    # 随机生成输入数据，形状为 [B, S, D]
    input_data = torch.randn(batch_size, feature_dim, sequence_length)

    # 定义WTConv1d的参数
    in_channels = feature_dim
    out_channels = feature_dim
    kernel_size = 5
    wt_levels = 2  # 小波分解的层数

    # 实例化 WTConv1d
    wtconv1d_layer = WTConv1d(in_channels, out_channels, kernel_size=kernel_size, wt_levels=wt_levels)

    multi_wtconv1d_layer = MultiLayerWTConv1d(num_layers=3, in_channels=6, out_channels=6, kernel_size=7, wt_levels=4)

    # 前向传播，计算输出
    # output_data = wtconv1d_layer(input_data)

    output_data = multi_wtconv1d_layer(input_data)

    # 打印输出的形状
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output_data.shape}")