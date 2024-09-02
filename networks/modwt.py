import numpy as np
import pdb
import pywt
import torch

def upArrow_op(li, j):
    if j == 0:
        return [1]
    N = len(li)
    li_n = np.zeros(2 ** (j - 1) * (N - 1) + 1)
    for i in range(N):
        li_n[2 ** (j - 1) * i] = li[i]
    return li_n


def period_list(li, N):
    n = len(li)
    # append [0 0 ...]
    n_app = N - np.mod(n, N)
    li = list(li)
    li = li + [0] * n_app
    if len(li) < 2 * N:
        return np.array(li)
    else:
        li = np.array(li)
        li = np.reshape(li, [-1, N])
        li = np.sum(li, axis=0)
        return li


def circular_convolve_mra( signal, ker ):
    '''
        signal: real 1D array
        ker: real 1D array
        signal and ker must have same shape
        Modification of 
            https://stackoverflow.com/questions/35474078/python-1d-array-circular-convolution
    '''
    return np.flip(np.real(np.fft.ifft( np.fft.fft(signal)*np.fft.fft(np.flip(ker))))).astype(np.int).tolist()


# def circular_convolve_d(h_t, v_j_1, j):
#     '''
#     jth level decomposition
#     h_t: \tilde{h} = h / sqrt(2)
#     v_j_1: v_{j-1}, the (j-1)th scale coefficients
#     return: w_j (or v_j)
#     '''
#     N = len(v_j_1)
#     L = len(h_t)
#     w_j = np.zeros(N)
#     l = np.arange(L)
#     for t in range(N):
#         index = np.mod(t - 2 ** (j - 1) * l, N)
#         v_p = np.array([v_j_1[ind] for ind in index])
#         w_j[t] = (np.array(h_t) * v_p).sum()
#     return w_j

# def circular_convolve_d(h_t, v_j_1, level):
#     filter_len = len(h_t)
#     N = len(v_j_1)
#     w_j = np.zeros((N, 30, 6), dtype=np.float64)  # 更新输出形状以匹配输入形状
    
#     # 将 v_j_1 从 GPU 内存复制到主机内存
#     if isinstance(v_j_1, torch.Tensor):
#         v_j_1 = v_j_1.cpu().numpy()
    
#     # 扩展 v_j_1 以支持循环卷积
#     extended_v = np.concatenate((v_j_1[-filter_len + 1:], v_j_1, v_j_1[:filter_len - 1]))

#     # 调整 h_t 的形状以适应多维数据
#     h_t = np.reshape(h_t, (1, 1, -1))
    
#     # 应用滤波器
#     for t in range(N):
#         v_p = extended_v[t:t + filter_len]  # 选择正确的循环片段
#         w_j[t] = np.tensordot(h_t, v_p, axes=([2], [0]))  # 应用多维卷积

#     return w_j

def circular_convolve_d(h_t, v_j_1, level):
    filter_len = len(h_t)
    N = len(v_j_1)
    device = v_j_1.device  # 保存原始设备信息

    # 确保所有输入都在CPU上，以便使用NumPy处理
    if isinstance(v_j_1, torch.Tensor):
        v_j_1 = v_j_1.cpu().numpy()

    if isinstance(h_t, torch.Tensor):
        h_t = h_t.cpu().numpy()  # 确保h_t也在CPU上

    # 扩展 v_j_1 以支持循环卷积
    extended_v = np.concatenate((v_j_1[-filter_len + 1:], v_j_1, v_j_1[:filter_len - 1]))

    # 调整 h_t 的形状以适应多维数据
    h_t = np.reshape(h_t, (1, 1, -1))

    w_j = np.zeros((N, 30, 6), dtype=np.float64)  # 更新输出形状以匹配输入形状
    for t in range(N):
        v_p = extended_v[t:t + filter_len]
        w_j[t] = np.tensordot(h_t, v_p, axes=([2], [0]))  # 应用多维卷积

    # 将结果转换回PyTorch张量，并移回原始设备
    return torch.from_numpy(w_j).to(device)





def circular_convolve_s(h_t, g_t, w_j, v_j, j):
    '''
    (j-1)th level synthesis from w_j, w_j
    see function circular_convolve_d
    '''
    N = len(v_j)
    L = len(h_t)
    v_j_1 = np.zeros(N)
    l = np.arange(L)
    for t in range(N):
        index = np.mod(t + 2 ** (j - 1) * l, N)
        w_p = np.array([w_j[ind] for ind in index])
        v_p = np.array([v_j[ind] for ind in index])
        v_j_1[t] = (np.array(h_t) * w_p).sum()
        v_j_1[t] = v_j_1[t] + (np.array(g_t) * v_p).sum()
    return v_j_1


# def modwt(x, filters, level):
#     '''
#     filters: 'db1', 'db2', 'haar', ...
#     return: see matlab
#     '''
#     # filter
#     wavelet = pywt.Wavelet(filters)
#     h = wavelet.dec_hi
#     g = wavelet.dec_lo
#     h_t = np.array(h) / np.sqrt(2)
#     g_t = np.array(g) / np.sqrt(2)
#     wavecoeff = []
#     v_j_1 = x
#     for j in range(level):
#         w = circular_convolve_d(h_t, v_j_1, j + 1)
#         v_j_1 = circular_convolve_d(g_t, v_j_1, j + 1)
#         wavecoeff.append(w)
#     wavecoeff.append(v_j_1)
#     return np.vstack(wavecoeff)


def modwt(x, filters, level):
    '''
    使用 PyTorch 实现的小波变换。
    filters: 'db1', 'db2', 'haar', ...
    x: 输入张量，应当已经是一个 torch.Tensor。
    level: 小波变换的层数。
    return: 各层小波系数堆叠的结果。
    '''
    wavelet = pywt.Wavelet(filters)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    
    # 将滤波器转换为张量并调整尺寸
    h_t = torch.tensor(h, dtype=torch.float32) / torch.sqrt(torch.tensor(2.0))
    g_t = torch.tensor(g, dtype=torch.float32) / torch.sqrt(torch.tensor(2.0))
    
    # 确保滤波器在正确的设备上
    device = x.device
    h_t = h_t.to(device)
    g_t = g_t.to(device)
    
    wavecoeff = []
    v_j_1 = x
    for j in range(level):
        w = circular_convolve_d(h_t, v_j_1, j + 1)
        v_j_1 = circular_convolve_d(g_t, v_j_1, j + 1)
        wavecoeff.append(w)
    wavecoeff.append(v_j_1)
    
    # 使用 PyTorch 的 stack 来合并结果，替代 np.vstack
    return torch.stack(wavecoeff, dim=0)  # 检查 dim 参数以确保结果的维度正确




def imodwt(w, filters):
    ''' inverse modwt '''
    # filter
    wavelet = pywt.Wavelet(filters)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    h_t = np.array(h) / np.sqrt(2)
    g_t = np.array(g) / np.sqrt(2)
    level = len(w) - 1
    v_j = w[-1]
    for jp in range(level):
        j = level - jp - 1
        v_j = circular_convolve_s(h_t, g_t, w[j], v_j, j + 1)
    return v_j


def modwtmra(w, filters):
    ''' Multiresolution analysis based on MODWT'''
    # filter
    wavelet = pywt.Wavelet(filters)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    # D
    level, N = w.shape
    level = level - 1
    D = []
    g_j_part = [1]
    for j in range(level):
        # g_j_part
        g_j_up = upArrow_op(g, j)
        g_j_part = np.convolve(g_j_part, g_j_up)
        # h_j_o
        h_j_up = upArrow_op(h, j + 1)
        h_j = np.convolve(g_j_part, h_j_up)
        h_j_t = h_j / (2 ** ((j + 1) / 2.))
        if j == 0: h_j_t = h / np.sqrt(2)
        h_j_t_o = period_list(h_j_t, N)
        D.append(circular_convolve_mra(h_j_t_o, w[j]))
    # S
    j = level - 1
    g_j_up = upArrow_op(g, j + 1)
    g_j = np.convolve(g_j_part, g_j_up)
    g_j_t = g_j / (2 ** ((j + 1) / 2.))
    g_j_t_o = period_list(g_j_t, N)
    S = circular_convolve_mra(g_j_t_o, w[-1])
    D.append(S)
    return np.vstack(D)


if __name__ == '__main__':
    s1 = np.arange(10)
    ws = modwt(s1, 'db2', 3)
    s1p = imodwt(ws, 'db2')
    mra = modwtmra(ws, 'db2')
