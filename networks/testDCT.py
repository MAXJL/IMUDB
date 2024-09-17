import torch
import torch_dct as dct

# 创建一个随机张量来模拟数据
original_data = torch.randn(1, 10, 128)  # 假设有1个batch，10个通道，128个数据点

# 应用DCT变换
transformed_data = dct.dct(original_data, norm='ortho')

# 应用逆DCT变换
recovered_data = dct.idct(transformed_data, norm='ortho')

# 比较原始数据和恢复后的数据
difference = torch.abs(original_data - recovered_data).mean()  # 计算平均绝对误差

print("Original Data: \n", original_data)
print("Recovered Data: \n", recovered_data)
print("Difference: \n", difference)