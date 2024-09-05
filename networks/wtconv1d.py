import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import pywt
import pywt.data
import torch
import torch.nn as nn
import numpy as np
from modwt import modwt, imodwt



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
    print(f"Decomposition filter shape: {dec_filters.shape}")
    print(f"Reconstruction filter shape: {rec_filters.shape}")
    return dec_filters, rec_filters

# def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
#     w = pywt.Wavelet(wave)
#     dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
#     dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)

#     # Decomposition filters
#     dec_filters = torch.stack([dec_lo, dec_hi], dim=0)  # Shape: [2, filter_size]
#     dec_filters = dec_filters.repeat(in_size, 1, 1)  # Repeat for each channel
#     dec_filters = dec_filters.view(in_size, 2, -1)  # Shape: [in_size, 2, filter_size]
#     dec_filters = dec_filters.permute(1, 0, 2).reshape(in_size * 2, 1, -1)  # Shape: [in_size*2, 1, filter_size]

#     # Reconstruction filters
#     rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type)
#     rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type)
#     rec_filters = torch.stack([rec_lo, rec_hi], dim=0)  # Shape: [2, filter_size]
#     rec_filters = rec_filters.repeat(out_size, 1, 1)  # Repeat for each channel
#     rec_filters = rec_filters.view(out_size, 2, -1)  # Shape: [out_size, 2, filter_size]
#     rec_filters = rec_filters.permute(1, 0, 2).reshape(out_size * 2, 1, -1)  # Shape: [out_size*2, 1, filter_size]

#     # print(f"Adjusted Decomposition filter shape: {dec_filters.shape}")
#     # print(f"Adjusted Reconstruction filter shape: {rec_filters.shape}")

#     return dec_filters, rec_filters




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

    # filters = filters.repeat(c // filters.shape[0], 1, 1)
    filters = filters.repeat(c // (filters.shape[0] // 2), 1, 1)
    x = F.conv_transpose1d(x, filters, stride=2, groups=c, padding=pad)
    return x


def w_transform(x, wavelet='db1', level=1, mode='symmetric'):
    # x: [batch_size, channels, seq_length]
    device = x.device
    batch_size, sequence_length, num_features = x.shape
    print(f"Input shape is {x.shape}")
    # Assuming each dwt operation halves the sequence length
    # concat_freq_components = np.zeros((batch_size, sequence_length // 2, num_features * 2))
    concat_freq_components = torch.zeros((batch_size, sequence_length // 2, num_features * 2))
    for i in range(batch_size):
        for j in range(num_features):
            x_np = x.detach().cpu().numpy()
            cA, cD = pywt.dwt(x_np[i, :, j], wavelet, mode=mode)  # Perform DWT
            # Concatenate low and high frequency components along the last dimension
            cA_tensor = torch.tensor(cA, device=device)
            cD_tensor = torch.tensor(cD, device=device)
            concat_freq_components[i, :, 2 * j] = cA_tensor
            concat_freq_components[i, :, 2 * j + 1] = cD_tensor
    
    # Convert the numpy array back to a torch tensor
    concat_freq_components = torch.tensor(concat_freq_components)
    print(f"concat_freq_components shape is {concat_freq_components.shape}")#[1024, 15, 12]
    return concat_freq_components

def inverse_wavelet_transform(combined_freq_data, wavelet='db1'):
    """
    Perform inverse wavelet transform on combined low and high frequency data.
    
    Args:
    combined_freq_data (numpy.array): Combined low and high frequency data of shape (batch_size, seq_len, num_combined_features)
    wavelet (str): Type of wavelet used in forward transformation.
    
    Returns:
    numpy.array: Reconstructed original data of shape (batch_size, original_seq_length, num_features)
    """
    if isinstance(combined_freq_data, torch.Tensor):
        combined_freq_data = combined_freq_data.cpu().numpy()  # 将 PyTorch 张量转换为 NumPy 数组
        
    batch_size, seq_len, num_combined_features = combined_freq_data.shape
    num_features = num_combined_features // 2
    original_seq_length = seq_len * 2
    reconstructed_data = np.zeros((batch_size, original_seq_length, num_features))
    
    for i in range(batch_size):
        for j in range(num_features):
            low_freq = combined_freq_data[:, :, 2*j]   # Extract low frequency components
            high_freq = combined_freq_data[:, :, 2*j+1] # Extract high frequency components
            # Perform the inverse DWT for each feature
            reconstructed_data[i, :, j] = pywt.idwt(low_freq[i, :], high_freq[i, :], wavelet)
    
    return torch.tensor(reconstructed_data)







class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class MultiLayerWTConv1d(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, kernel_size=3, wt_levels=2):
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
    
class CustomCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rates, dropout):
        super(CustomCNN, self).__init__()
        c0 = 16
        c1 = 2 * c0
        c2 = 2 * c1
        c3 = 2 * c2
        p0 = (kernel_size-1) + dilation_rates[0]*(kernel_size-1) + dilation_rates[0]*dilation_rates[1]*(kernel_size-1) + dilation_rates[0]*dilation_rates[1]*dilation_rates[2]*(kernel_size-1)
        self.cnn = nn.Sequential(
            nn.ReplicationPad1d((p0, 0)),  # padding at start
            nn.Conv1d(in_channels, c0, kernel_size, dilation=1),
            nn.BatchNorm1d(c0),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(c0, c1, kernel_size, dilation=dilation_rates[0]),
            nn.BatchNorm1d(c1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(c1, c2, kernel_size, dilation=dilation_rates[0]*dilation_rates[1]),
            nn.BatchNorm1d(c2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(c2, c3, kernel_size, dilation=dilation_rates[0]*dilation_rates[1]*dilation_rates[2]),
            nn.BatchNorm1d(c3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(c3, out_channels, 1, dilation=1)
        )
    def forward(self, x):
        return self.cnn(x)

class IntegratedWTConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dropout=0.5, stride=1, bias=True, wt_levels=2, wt_type='db1', dilation_rates=[4, 4, 4]):
        super(IntegratedWTConv1d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation_rates = dilation_rates

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform_1d, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform_1d, filters=self.iwt_filter)


        self.cnn = CustomCNN(in_channels, in_channels, kernel_size, dilation_rates, dropout)

        self.base_scale = _ScaleModule([1, in_channels, 1])
        wavelet_out_channels = in_channels * 2

        # 将 GyroNet 的卷积结构替换为 wavelet_convs
        self.wavelet_convs = nn.ModuleList(
            [CustomCNN(wavelet_out_channels, wavelet_out_channels, kernel_size, dilation_rates, dropout) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, wavelet_out_channels, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )
        # 条件处理 stride
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
            if (curr_shape[2] % 2 > 0):
                curr_pads = (0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0]
            
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 2, shape_x[3])
            
            # 使用GyroNet中的卷积结构替换原有的Conv1d操作
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0])
            x_h_in_levels.append(curr_x_tag[:, :, 1])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h.unsqueeze(2)], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(x)

        x = x + x_tag

        # 如果 stride > 1，则进行 stride 操作
        if self.do_stride is not None:
            x = self.do_stride(x)

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

            curr_x = self.wt_function(curr_x_ll) #curr_x:[1024, 6, 2, 15]
            curr_x_ll = curr_x[:,:,0] #curr_x_ll:[1024, 6, 15] 提取低频信号
            
            shape_x = curr_x.shape #shape_x:[1024, 6, 2, 15]
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 2, shape_x[3]) #curr_x_tag:[1024, 12, 15]
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag)) #curr_x_tag:[1024, 12, 15]
            curr_x_tag = curr_x_tag.reshape(shape_x) #curr_x_tag:[1024, 6, 2, 15]

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


class W_Transform(nn.Module):
    def __init__(self, wavelet='db1', level=1, mode='symmetric'):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.mode = mode

    def forward(self, x):
        # x: [batch_size, channels, seq_length]
        batch_size, sequence_length, num_features = x.shape
        print(f"Input shape is {x.shape}")
        # Assuming each dwt operation halves the sequence length
        concat_freq_components = np.zeros((batch_size, sequence_length // 2, num_features * 2))
        
        for i in range(batch_size):
            for j in range(num_features):
                x_np = x.detach().cpu().numpy()
                cA, cD = pywt.dwt(x_np[i, :, j], self.wavelet, mode=self.mode)  # Perform DWT
                # print(f"cA shape is {cA.shape}, cD shape is {cD.shape}")
                # Concatenate low and high frequency components along the last dimension
                concat_freq_components[i, :, 2 * j] = cA
                concat_freq_components[i, :, 2 * j + 1] = cD
                # print(f"IN LOOP:concat_freq_components shape is {concat_freq_components.shape}")
        
        # Convert the numpy array back to a torch tensor
        concat_freq_components = torch.tensor(concat_freq_components)
        print(f"concat_freq_components shape is {concat_freq_components.shape}")
        
        return concat_freq_components

if __name__ == '__main__':
    # 定义输入的参数
    batch_size = 1024  # Batch size
    sequence_length = 30  # Sequence length (S)
    feature_dim = 6  # Feature dimension (D)

    # 随机生成输入数据，形状为 [B, S, D]
    input_data = torch.randn(batch_size, sequence_length, feature_dim)

    # 定义WTConv1d的参数
    in_channels = feature_dim
    out_channels = feature_dim
    kernel_size = 3
    wt_levels = 2  # 小波分解的层数
    dilation_rates=[2, 2, 2]


    # # 实例化 WTConv1d
    # wtconv1d_layer = WTConv1d(in_channels, out_channels, kernel_size=3, wt_levels=2)

    # multi_wtconv1d_layer = MultiLayerWTConv1d(num_layers=3, in_channels=6, out_channels=6, kernel_size=3, wt_levels=2)

    # #实例化IntegratedWTConv1d

    # integrated_wtconv1d_layer = IntegratedWTConv1d(in_channels=6, out_channels=6, kernel_size=5, wt_levels=2, dilation_rates=dilation_rates)
    

    W_Transform_layer = W_Transform(wavelet='db1', level=1)


    # 前向传播，计算输出
    #计算推理时间
    import time
    start = time.time()
    # output_data = wtconv1d_layer(input_data)
    # output_data = multi_wtconv1d_layer(input_data)
    # output_data = integrated_wtconv1d_layer(input_data)
    output_data = W_Transform_layer(input_data)

    data = inverse_wavelet_transform(output_data)

    end = time.time()
    print(f"Time: {end - start}")

    # 打印输出的形状
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output_data.shape}")
    print(f"Data shape: {data.shape}")
    