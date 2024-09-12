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
    # filters = filters.repeat(c // filters.shape[0], 1, 1)
    filters = filters.repeat(c // (filters.shape[0] // 2), 1, 1)
    x = F.conv_transpose1d(x, filters, stride=2, groups=c, padding=pad)

    # 重新排列时间步，每两个元素交换位置
    # 创建一个索引映射
    indices = torch.arange(x.shape[-1])
    # 适用于偶数长度的序列
    indices = indices.view(-1, 2).flip(dims=[1]).reshape(-1)
    x = x.index_select(2, indices.to(device))

    return x


##########################################

# def create_wavelet_filter_1d(wave, in_channels, type=torch.float):
#     w = pywt.Wavelet(wave)
#     dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
#     dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
#     # 创建一维小波核
#     dec_filters = torch.stack([dec_lo, dec_hi], dim=0).view(2, 1, -1).repeat(in_channels, 1, 1)
#     rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type)
#     rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type)
#     # 创建一维逆小波核
#     rec_filters = torch.stack([rec_lo, rec_hi], dim=0).view(2, 1, -1).repeat(in_channels, 1, 1)
#     return dec_filters, rec_filters
# def wavelet_transform_1d(data, filters):
#     """
#     一维小波变换
#     """
#     device = data.device
#     filters = filters.to(device)
#     # 这里stride=2用于下采样，只沿着序列长度操作
#     transformed_data = F.conv1d(data, filters, stride=2, padding=filters.shape[-1]//2 - 1, groups=data.shape[1])
#     b, c, s = transformed_data.shape
#     transformed_data = transformed_data.view(b, c//2, 2, s).permute(0, 1, 3, 2).reshape(b, c//2 * 2, s)
#     return transformed_data

# def inverse_wavelet_transform_1d(transformed_data, filters):
#     device = transformed_data.device
#     filters = filters.to(device)
#     b, c, s = transformed_data.shape
#     transformed_data = transformed_data.view(b, c//2, 2, s).permute(0, 1, 3, 2).reshape(b, c, s)
#     original_data = F.conv_transpose1d(transformed_data, filters, stride=2, padding=filters.shape[-1]//2 - 1, groups=transformed_data.shape[1]//2)
#     return original_data


###########################################################################












def w_transform(x, wavelet='db1', level=1, mode='symmetric'):
    # x: [batch_size, channels, seq_length]
    device = x.device
    batch_size, sequence_length, num_features = x.shape
    concat_freq_components = torch.zeros((batch_size, sequence_length // 2, num_features * 2))
    for i in range(batch_size):
        for j in range(num_features):
            cA, cD = pywt.dwt(x[i, :, j], wavelet, mode=mode)
            cA_tensor = torch.tensor(cA, device=device)
            cD_tensor = torch.tensor(cD, device=device)
            concat_freq_components[i, :, j] = cA_tensor
            concat_freq_components[i, :, num_features + j] = cD_tensor
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
    # if isinstance(transformed_data, torch.Tensor):
    #     transformed_data = transformed_data.cpu().numpy()  # 将 PyTorch 张量转换为 NumPy 数组
    transformed_data = combined_freq_data
    batch_size, seq_len, num_combined_features = transformed_data.shape
    num_features = num_combined_features // 2
    original_seq_length = seq_len * 2
    reconstructed_data = np.zeros((batch_size, original_seq_length, num_features))
    
    for i in range(batch_size):
        for j in range(num_features):
            low_freq = transformed_data[i, :, j]   # 提取低频分量
            high_freq = transformed_data[i, :, num_features + j] # 提取高频分量
            # 对每个特征执行逆小波变换
            reconstructed_data[i, :, j] = pywt.idwt(low_freq, high_freq, wavelet)
    
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



##############自定义的，出现了变量梯度计算的问题################################

# def wavelet_transform(x, wavelet='db1', level=1, mode='symmetric'):
#     device = x.device
#     batch_size, sequence_length, num_features = x.shape
#     concat_freq_components = torch.zeros((batch_size, sequence_length // 2, num_features * 2), device=device)
#     for i in range(batch_size):
#         for j in range(num_features):
#             x_slice = x[i, :, j].detach().cpu().numpy()
#             cA, cD = pywt.dwt(x_slice, wavelet, mode=mode)
#             cA_tensor = torch.tensor(cA, device=device)
#             cD_tensor = torch.tensor(cD, device=device)
#             concat_freq_components[i, :, 2 * j] = cA_tensor
#             concat_freq_components[i, :, 2 * j + 1] = cD_tensor
#     return concat_freq_components

def wavelet_transform(x, wavelet='db1', wtlevel=1, mode='symmetric'):
    device = x.device
    batch_size, sequence_length, num_features = x.shape
    # The output dimensions will be adjusted based on the level of decomposition
    output_length = sequence_length // (2 ** wtlevel)
    num_output_features = num_features * (1 + wtlevel)
    concat_freq_components = torch.zeros((batch_size, output_length, num_output_features), device=device)
    
    for i in range(batch_size):
        for j in range(num_features):
            x_slice = x[i, :, j].detach().cpu().numpy()
            coeffs = pywt.wavedec(x_slice, wavelet, mode=mode, level=wtlevel)
            for k in range(len(coeffs)):
                coeffs_tensor = torch.tensor(coeffs[k], device=device)
                concat_freq_components[i, :len(coeffs[k]), j * (1 + wtlevel) + k] = coeffs_tensor
    
    return concat_freq_components

def wavelet_inverse_transform(combined_freq_data, wavelet='haar', wtlevel=1):
    device = combined_freq_data.device
    batch_size, seq_len, num_combined_features = combined_freq_data.shape
    num_features = num_combined_features // (1 + wtlevel)
    original_seq_length = seq_len * (2 ** wtlevel)
    reconstructed_data = torch.zeros((batch_size, original_seq_length, num_features), device=device)
    
    for i in range(batch_size):
        for j in range(num_features):
            coeffs = []
            for k in range(wtlevel + 1):
                coeff_len = seq_len * (2 ** k) if k < wtlevel else original_seq_length
                coeffs.append(combined_freq_data[i, :coeff_len, j * (1 + wtlevel) + k].detach().cpu().numpy())
            reconstructed_np = pywt.waverec(coeffs, wavelet)
            reconstructed_data[i, :, j] = torch.tensor(reconstructed_np[:original_seq_length], device=device)
    
    return reconstructed_data

############################################################################

 


def modwt(x, filters, level):
    device = x.device
    batch_size, time_steps, features = x.shape
    wavelet = pywt.Wavelet(filters)
    h = torch.tensor(wavelet.dec_hi[::-1], dtype=torch.float32, device=device) / torch.sqrt(torch.tensor(2.0,device=device))
    g = torch.tensor(wavelet.dec_lo[::-1], dtype=torch.float32, device=device) / torch.sqrt(torch.tensor(2.0,device=device))
    wavecoeff = []
    v_j_1 = x.to(device)
    for j in range(level):
        w = circular_convolve_d(h, v_j_1, j + 1)
        v_j_1 = circular_convolve_d(g, v_j_1, j + 1)
        wavecoeff.append(w)
    wavecoeff.append(v_j_1)
    wavecoeff = torch.stack(wavecoeff, dim=-1)  # 将级数维度放到最后
    wavecoeff = wavecoeff.reshape(batch_size, time_steps, -1)  # 合并级数维度到特征维度
    return wavecoeff

def imodwt(w, filters, level):
    device = w.device
    batch_size, time_steps, features = w.shape
    features = features // (level + 1)  # 恢复原始特征数
    # w = w.reshape(batch_size, time_steps, features, level + 1).transpose(0, 3, 1, 2)  # 重新调整维度
    w = w.view(batch_size, time_steps, features, level + 1).permute(0, 3, 1, 2)  # 重新调整维度
    wavelet = pywt.Wavelet(filters)
    h = torch.tensor(wavelet.dec_hi[::-1], dtype=torch.float32,device=device) / torch.sqrt(torch.tensor(2.0,device=device))
    g = torch.tensor(wavelet.dec_lo[::-1], dtype=torch.float32,device=device) / torch.sqrt(torch.tensor(2.0,device=device))
    v_j = w[:, -1, :, :]  # 初始化为最后一级的逼近系数
    for jp in range(level):
        j = level - jp - 1
        v_j = circular_convolve_s(h, g, w[:, j, :, :], v_j, j + 1)
    return v_j

def circular_convolve_d(h_t, v_j_1, j):
    device = v_j_1.device
    batch_size, N, features = v_j_1.shape
    L = len(h_t)
    # w_j = np.zeros_like(v_j_1)
    # l = np.arange(L)
    w_j = torch.zeros_like(v_j_1)
    l = torch.arange(L, device=device)
    for t in range(N):
        # index = np.mod(t - 2 ** (j - 1) * l, N)
        index = torch.remainder(t - 2 ** (j - 1) * l, N)
        v_p = v_j_1[:, index, :]
        # w_j[:, t, :] = (h_t[:, None] * v_p).sum(axis=1)
        w_j[:, t, :] = torch.sum(h_t[None, :, None].to(device) * v_p, dim=1)
    return w_j

def circular_convolve_s(h_t, g_t, w_j, v_j, j):
    device = v_j.device
    batch_size, N, features = v_j.shape
    L = len(h_t)
    # v_j_1 = np.zeros_like(v_j)
    # l = np.arange(L)
    v_j_1 = torch.zeros_like(v_j)
    l = torch.arange(L,device=device)
    for t in range(N):
        # index = np.mod(t + 2 ** (j - 1) * l, N)
        index = torch.remainder(t + 2 ** (j - 1) * l, N)
        w_p = w_j[:, index, :]
        v_p = v_j[:, index, :]
        # v_j_1[:, t, :] = (h_t[:, None] * w_p).sum(axis=1) + (g_t[:, None] * v_p).sum(axis=1)
        v_j_1[:, t, :] = torch.sum(h_t[None, :, None].to(device) * w_p, dim=1) + torch.sum(g_t[None, :, None].to(device) * v_p, dim=1)
    return v_j_1




if __name__ == '__main__':
    # 定义输入的参数
    batch_size = 1024  # Batch size
    sequence_length = 32  # Sequence length (S)
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
    wtconv1d_layer = WTConv1d(in_channels, out_channels, kernel_size=3, wt_levels=2)

#     # multi_wtconv1d_layer = MultiLayerWTConv1d(num_layers=3, in_channels=6, out_channels=6, kernel_size=3, wt_levels=2)

#     # #实例化IntegratedWTConv1d

#     # integrated_wtconv1d_layer = IntegratedWTConv1d(in_channels=6, out_channels=6, kernel_size=5, wt_levels=2, dilation_rates=dilation_rates)
    

#     W_Transform_layer = W_Transform(wavelet='db1', level=1)


    # 前向传播，计算输出
    #计算推理时间
    # import time
    # start = time.time()
    # output_data = wtconv1d_layer(input_data)
    # # output_data = multi_wtconv1d_layer(input_data)
    # # output_data = integrated_wtconv1d_layer(input_data)
    # # output_data = W_Transform_layer(input_data)

    # data = inverse_wavelet_transform(output_data)

    # end = time.time()
    # print(f"Time: {end - start}")

    # # 打印输出的形状
    # print(f"Input shape: {input_data.shape}")
    # print(f"Output shape: {output_data.shape}")
    # print(f"Data shape: {data.shape}")
    



#     print(f"Start testing the wavelet transform and inverse transform functions")
#      # Initialize a sample tensor
#   # Define a simple dataset
#     # data = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]])



    # data = torch.rand(1, 6, 4)


    # # Define wavelet type and level
    # wavelet_type = 'db1'
    # in_channels = out_channels = 6
    # levels = 2
    # # Create filters
    # dec_filters, rec_filters = create_wavelet_filter(wavelet_type, in_channels, out_channels)
    # # Perform wavelet transform
    # wt_output = wavelet_transform_1d(data, dec_filters)
    # # Perform inverse wavelet transform
    # iwt_output = inverse_wavelet_transform_1d(wt_output, rec_filters)
    # # Print the results
    # # print("Original data:")
    # print("原始数据的维度", data.shape)
    # print("原始数据:")
    # print(data)
    # print("分解后的维度:", wt_output.shape)
    # # print("Inverse wavelet transform output (reconstructed data):")
    # # print("分解后的数据:", wt_output)
    # # print(iwt_output)
    # print("重构后的维度:", iwt_output.shape)
    # print("重构后的数据:")
    # print(iwt_output)
    # # 验证重构后的信号是否与原信号相同，添加阈值
    # atol = 1e-6  # 绝对误差阈值
    # rtol = 1e-5  # 相对误差阈值
    # print("重构是否与原信号相同:", np.allclose(data, iwt_output, atol=atol, rtol=rtol))



    # Sample data creation
    # batch_size = 1
    # num_channels = 6
    # sequence_length = 10  # Ensure sequence length is even for simplicity
    # data = torch.randn(batch_size, num_channels, sequence_length)
    # #torch创建一个维度是【1，6，10】的tensor,1是batch_size,6是通道数，10是序列长度
    
    # # Define wavelet type and create filters
    # wavelet_type = 'db1'
    # dec_filters, rec_filters = create_wavelet_filter_1d(wavelet_type, num_channels)

    # print("Original data shape:", data.shape)
    # print("Original data:", data)
    # # Perform wavelet transform
    # wt_output = wavelet_transform_1d(data, dec_filters)
    # print("Wavelet transform output shape:", wt_output.shape)
    # # Perform inverse wavelet transform
    # iwt_output = inverse_wavelet_transform_1d(wt_output, rec_filters)
    # print("Inverse wavelet transform output shape:", iwt_output.shape)
    # print("Inverse wavelet transform output:", iwt_output)

    # atol = 1e-6  # 绝对误差阈值
    # rtol = 1e-5  # 相对误差阈值
    # print("重构是否与原信号相同:", np.allclose(data, iwt_output, atol=atol, rtol=rtol))












    # # 示例验证 形状为 (1024, 30, 6) 的tensor数据
    # x = torch.rand(1, 20, 6)
    # # x = np.random.rand(1024, 30, 6)  # 形状为 (1024, 30, 6) 的随机数组
  
    # # 小波滤波器
    # filters = 'haar'

    # # 分解级别
    # level = 2

    # # 调用 modwt 函数
    # wavecoeff = modwt(x, filters, level)


    # # 调用 imodwt 函数
    # reconstructed_x = imodwt(wavecoeff, filters, level)

    # # 输出维度
    # print("输入维度:", x.shape)
    # print("原始数据", x)
    # print("分解后的维度:", wavecoeff.shape)
    # # print("分解后的数据", wavecoeff)
    # print("重构后的维度:", reconstructed_x.shape)
    # print("重构后的数据", reconstructed_x)

    # # 验证重构后的信号是否与原信号相同，添加阈值
    # atol = 1e-6  # 绝对误差阈值
    # rtol = 1e-5  # 相对误差阈值
    # print("重构是否与原信号相同:", np.allclose(x, reconstructed_x, atol=atol, rtol=rtol))





    # # 创建一个模拟的Tensor数据，形状为[1024, 30, 6]
    # data_tensor = torch.rand(1024, 30, 6)

    # # wavelet = 'db1'
    # # 执行小波变换
    # transformed_tensor = wavelet_transform(data_tensor)

    # # 使用定义的 wavelet_inverse_transform 函数进行逆变换
    # reconstructed_tensor = wavelet_inverse_transform(transformed_tensor)

    # # 验证原始数据和重构数据是否一致
    # if torch.allclose(data_tensor, reconstructed_tensor, atol=1e-5):
    #     print("检查结果：原始数据与重构数据相同。")
    # else:
    #     print("检查结果：原始数据与重构数据不相同。")

    # print("原始Tensor形状:", data_tensor.shape)
    # print("变换后的Tensor形状:", transformed_tensor.shape)
    # print("重构后的Tensor形状:", reconstructed_tensor.shape)


    # data = torch.rand(1, 4, 6)
    # print("before wavelet shape:",data.shape)
    # print("原序列：",data)
    # data = wavelet_transform(data,wtlevel=2)
    # print("after wavelet shape:",data.shape)
    # print(data)
    # inverse_data = wavelet_inverse_transform(data,wtlevel=2)
    # print("inverse_data shape:",inverse_data.shape)
    # print("重构序列：",inverse_data)