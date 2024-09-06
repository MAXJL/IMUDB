import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from wtconv1d import WTConv1d
from wtconv1d import MultiLayerWTConv1d
from wtconv1d import IntegratedWTConv1d
from wtconv1d import SEBlock
from wtconv1d import W_Transform

from functools import partial
import pywt
import pywt.data
from wtconv1d import wavelet_transform_1d
from wtconv1d import inverse_wavelet_transform_1d
from wtconv1d import create_wavelet_filter

from wtconv1d import w_transform
from wtconv1d import inverse_wavelet_transform

from common.mask import create_mask

# from mLSTM import mLSTM
# from sLSTM import sLSTM


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = x.size(-1) // -np.prod(shape)
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    # assert n_dims > 1 and n_dims < len(s) # bad practice to include assert for production code, thus commented
    return x.view(*s[:-n_dims], -1)

# 每个元素减去其平均值并除以其标准差，最后使用可学习的参数gamma和beta进行缩放和位移
class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.hidden), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(cfg.hidden), requires_grad=True)
        # self.gamma = nn.Parameter(torch.ones(36), requires_grad=True)
        # self.beta = nn.Parameter(torch.zeros(36), requires_grad=True)
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class ContextAwareLayerNorm(nn.Module):
    """ Layer normalization that adapts to the input context. """
    def __init__(self, cfg):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.hidden), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(cfg.hidden), requires_grad=True)
        self.variance_epsilon = 1e-12

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        context_scale = torch.sigmoid(self.gamma * std + self.beta)  # Context-aware scaling
        return (x - mean) / (std + self.variance_epsilon) * context_scale





# 这个嵌入层的主要作用是将输入的特征向量转换为适合模型处理的嵌入表示，并添加位置信息以保留序列的顺序信息。具体来说，它通过以下步骤实现这一目标：
# 1线性变换：首先，通过线性层 self.lin 将输入特征向量 x 映射到隐藏层维度。这一步将原始特征转换为模型可以处理的嵌入表示。
# 2位置嵌入：为了保留序列中每个元素的位置信息，嵌入层使用位置嵌入 self.pos_embed。如果没有传入位置嵌入，则创建一个新的位置嵌入层。位置嵌入的大小为序列长度 cfg.seq_len 和隐藏层维度 cfg.hidden。
# 3归一化：如果配置中 emb_norm 为真，则对嵌入进行层归一化操作，以稳定训练过程并提高模型性能。
# 4位置编码加成：将位置嵌入加到嵌入表示上，使得每个位置的嵌入不仅包含特征信息，还包含位置信息。
# 5再次归一化：最后，再次对嵌入进行层归一化，确保输出的嵌入表示稳定且适合后续的模型处理。
# 通过这些步骤，嵌入层将输入特征向量转换为包含位置信息的嵌入表示，为后续的模型层提供了更丰富的信息。

class Embeddings(nn.Module):

    def __init__(self, cfg, pos_embed=None):
        super().__init__()

        # factorized embedding   cfg feature映射到hidden 
        self.lin = nn.Linear(cfg.feature_num, cfg.hidden)
        if pos_embed is None: #若没有传入pos_embed，则创建一个pos_embed
        
            self.pos_embed = nn.Embedding(cfg.seq_len, cfg.hidden) # position embedding
        else:
            self.pos_embed = pos_embed

        self.norm = LayerNorm(cfg) #初始化了一个LayerNorm模块
        self.emb_norm = cfg.emb_norm

    def forward(self, x):
        seq_len = x.size(1) #获取序列长度
        #创建一个表示位置的张量，大小为（B, S） B是batch_size，S是序列长度
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len) # (S,) -> (B, S)

        # factorized embedding
        #接下来通过线性层lin对输入x进行变换，得到嵌入e
        e = self.lin(x)
        #如果emb_norm为True，则对e进行LayerNorm归一化操作
        if self.emb_norm:
            e = self.norm(e)
        #将位置编码pos加到嵌入e上
        e = e + self.pos_embed(pos)
        # print("Embeddings e shape:", e.shape)
        #再次归一化
        return self.norm(e)


# 该模块包含了Transformer的核心组件，即多头自注意力机制和前馈神经网络。具体来说，Block模块包含以下几个步骤：
# 1. 多头自注意力机制：首先，通过MultiHeadedSelfAttention模块对输入进行多头自注意力机制操作，得到注意力输出。
# 2. 残差连接：将输入和注意力输出相加，得到残差连接结果。
# 3. 层归一化：对残差连接结果进行层归一化操作，得到归一化结果。
# 4. 前馈神经网络：对归一化结果进行前馈神经网络操作，得到前馈网络输出。
# 5. 残差连接：将前馈网络输出和归一化结果相加，得到残差连接结果。
# 6. 再次层归一化：对残差连接结果进行层归一化操作，得到最终输出。
# 通过这些步骤，Block模块实现了Transformer的核心组件，包括多头自注意力机制和前馈神经网络，为模型提供了强大的建模能力。

class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_k = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_v = nn.Linear(cfg.hidden, cfg.hidden)
        self.scores = None # for visualization
        self.n_heads = cfg.n_heads

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        scores = F.softmax(scores, dim=-1)
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        h = merge_last(h, 2)
        self.scores = scores
        # print("MultiHeadedSelfAttention h shape:", h.shape)
        return h


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

#position-wise feed-forward networks
#该模块包含了Transformer的另一个核心组件，即位置编码前馈神经网络。具体来说，PositionWiseFeedForward模块包含以下几个步骤：
# 1. 线性变换：首先，通过线性层 self.fc1 对输入进行线性变换，得到中间表示。
# 2. 激活函数：对中间表示进行激活函数操作，通常使用GELU激活函数。
# 3. 再次线性变换：对激活函数输出进行线性变换，得到最终输出。
# 通过这些步骤，PositionWiseFeedForward模块实现了位置编码前馈神经网络，为模型提供了更强大的建模能力。

class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden, cfg.hidden_ff)
        self.fc2 = nn.Linear(cfg.hidden_ff, cfg.hidden)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        # print("PositionWiseFeedForward x shape:", self.fc2(gelu(self.fc1(x))).shape)
        return self.fc2(gelu(self.fc1(x)))




class DynamicWeights(nn.Module): #动态权重 用于动态调整注意力头的权重 
    """ A module to dynamically adjust the weights of attention heads. """
    # 下面是用法示例：
    # self.attn = MultiHeadedSelfAttention(cfg)
    # self.adaptive_merge = AdaptiveMerge(cfg)  # Add adaptive merge
    # self.proj = nn.Linear(cfg.hidden, cfg.hidden)
    # h = self.attn(h)
    # h = self.adaptive_merge(h)

    def __init__(self, cfg):
        super().__init__()
        self.head_weights = nn.Parameter(torch.ones(cfg.n_heads), requires_grad=True)

    def forward(self, attn_outputs):
        # attn_outputs should have shape [batch_size, num_heads, seq_length, hidden_dim]
        # Applying dynamic weights to the outputs of each head
        weighted_outputs = attn_outputs * self.head_weights.view(1, -1, 1, 1)
        return weighted_outputs.sum(dim=1)  # Combine heads




class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings(cfg)
        # Original BERT not used parameter-sharing strategies
        # self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

        # To used parameter-sharing strategies
        self.n_layers = cfg.n_layers
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        # self.drop = nn.Dropout(cfg.p_drop_hidden)
    def forward(self, x):
        # print(x.shape
        h = self.embed(x)
    
        # print("Embedding output h shape:", h.shape)
        for _ in range(self.n_layers):
            # h = block(h, mask)
            #打印第几次循环
            # print("Transformer loop:", _)
            h = self.attn(h)
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h))
        return h


class WaveletTransformer(nn.Module):
    def __init__(self, cfg):

        super(WaveletTransformer, self).__init__()

        # self.wtconv1d = WTConv1d(
        #     in_channels=cfg.feature_num, 
        #     out_channels=cfg.feature_num, 
        #     kernel_size=5, 
        #     wt_levels=2,
        #     )

        self.multiwtconv1d = MultiLayerWTConv1d(
            in_channels=cfg.feature_num, 
            out_channels=cfg.feature_num, 
            kernel_size=7, 
            wt_levels=3,
            num_layers=3)
        

        
        # self.intergratedwtconv1d = IntegratedWTConv1d(
        #     in_channels=cfg.feature_num, 
        #     out_channels=cfg.feature_num, 
        #     kernel_size=5, 
        #     wt_levels=2,
        #     dropout=0.5,
        #     dilation_rates=[1, 2, 4])
        

        # self.se_block = SEBlock(cfg.feature_num)  # 配置 SEBlock

        self.transformer = Transformer(cfg)

    def forward(self, x):
        # print(f"Input shape before permute: {x.shape}")
        # 从 [1024, 30, 6] 转换为 [1024, 6, 30]
        x = x.permute(0, 2, 1)
        # print(f"Shape after permute to [1024, 6, 30]: {x.shape}")
        # 通过 WTConv1d 处理
        # x = self.wtconv1d(x)
        x = self.multiwtconv1d(x)
        # x = self.intergratedwtconv1d(x)

        # x = self.se_block(x)  # 通过 SEBlock 处理

        # 再从 [1024, 6, 30] 转换回 [1024, 30, 6]
        x = x.permute(0, 2, 1)
        # print(f"Shape after permute back to [1024, 30, 6]: {x.shape}")
        # 然后将处理后的数据传递给 Transformer
        x = self.transformer(x)
        return x



class WT_Inter_Transformer(nn.Module):
    def __init__(self, cfg, in_channels,out_channels,wt_levels=2, wt_type='db1'):
        super(WT_Inter_Transformer, self).__init__()

        self.transformer = Transformer(cfg)

        assert in_channels == out_channels
        self.in_channels = in_channels
        self.wt_levels = wt_levels

        self.wt_type = wt_type

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        # print(f"WT filter shape: {self.wt_filter.shape}")
        # print(f"IWT filter shape: {self.iwt_filter.shape}")
        self.wt_function = partial(wavelet_transform_1d, filters = self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform_1d, filters = self.iwt_filter) 
        # self.adaptive_conv = nn.Conv1d(in_channels=12, out_channels=6, kernel_size=1)

    def forward(self, x):
        # Step 1: 小波变换
        # x shape: [1024, 30, 6]
        #reshape x to [1024, 6, 30]
        x = x.permute(0, 2, 1)
        x = self.wt_function(x) #shape: b, c, 2, w // 2 [1024, 6, 2, 15]
        # x shape: [1024, 6, 2, 15]
        #reshape x to [1024, 12, 15]
        x = x.view(x.size(0), x.size(1) * 2, x.size(3))
        # reshape: [1024, 15, 12]  for transformer
        x = x.permute(0, 2, 1)

        # print("x shape after wt:", x.shape)

        # Step 2: Transformer
        x = self.transformer(x)
        # print("Transformer output h shape:", x.shape)
        #x shape: [1024, 15, 72]
        #reshape x for inverse wavelet transform
        #step 3: 准备小波逆变换 
        # batch_size, seq_length, feature_dim = x.shape # [1024, 15, 72]
        # # print("batch_size, seq_length, feature_dim:", batch_size, seq_length, feature_dim)
        # num_channels = feature_dim // 2 # 72 // 2 = 36
        # # # reshape x for inverse wavelet transform
        # x = x.permute(0, 2, 1) # [1024, 72, 15]
        # x = x.view(batch_size, num_channels, 2, seq_length) #[1024, 36, 2, 15]
        # # # Step 4: 小波逆变换
        # # print("x shape before iwt:", x.shape)
        # x = self.iwt_function(x)
        # # # x shape: [1024, 36, 30]
        # x = x.permute(0, 2, 1) # [1024, 30, 36]p
        # # print("x shape after iwt:", x.shape)


        return x


class W_Transform_without_inverseW(nn.Module):
    def __init__(self, cfg):
        super(W_Transform_without_inverseW, self).__init__()

        self.transformer = Transformer(cfg)

    def forward(self, x):
        
        device = x.device  # 获取输入张量所在的设备
        self.transformer.to(device)  # 将 transformer 模型移动到相同的设备

        x = w_transform(x, wavelet='db1') # [1024, 15, 12]
        # print("x shape after w_transform:", x.shape)


        x = self.transformer(x) # [1024, 15, 72]

        return x


    

class LIMUBertModel4Pretrain(nn.Module):

    def __init__(self, cfg, output_embed=False):
        super().__init__()

        self.transformer = Transformer(cfg) # encoder

        # self.wavelet_transformer = WaveletTransformer(cfg)

        # self.wt_inter_transformer = WT_Inter_Transformer(cfg, in_channels=6, out_channels=6, wt_levels=2, wt_type='db1')

        #self.W_Transform_without_inverseW = W_Transform_without_inverseW(cfg) 
      
        self.fc = nn.Linear(cfg.hidden, cfg.hidden)
        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        # self.linear = nn.Linear(36,36)

        self.activ = gelu
        self.norm = LayerNorm(cfg)

        self.norm_wt = LayerNorm2(cfg)

        self.decoder = nn.Linear(cfg.hidden, cfg.feature_num)
        # self.decoder = nn.Linear(cfg.hidden // 2, 6)

        self.output_embed = output_embed

        self.wtdecoder = nn.Linear(cfg.hidden, cfg.feature_num)
        # above is  transformer initialization

        wt_type = 'db1'
        in_channels = 6
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        # print(f"WT filter shape: {self.wt_filter.shape}")
        # print(f"IWT filter shape: {self.iwt_filter.shape}")
        self.wt_function = partial(wavelet_transform_1d, filters = self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform_1d, filters = self.iwt_filter) 


    def forward(self, input_seqs, masked_pos=None):

        device = input_seqs.device  # 获取输入张量所在的设备
        input_seqs = input_seqs.to(device)  # 确保输入张量在正确的设备上

        #####小波变换##########################################33
        # print("input_seqs shape:", input_seqs.shape)
        # input_seqs = w_transform(input_seqs, wavelet='db1') # [1024, 15, 12]
        # print("input_seqs shape after w_transform:", input_seqs.shape)
        # h_masked = self.wavelet_transformer(input_seqs)
        input_seqs = input_seqs.permute(0, 2, 1)
        input_seqs = self.wt_function(input_seqs) #shape: b, c, 2, w // 2 [1024, 6, 2, 15]
        # x shape: [1024, 6, 2, 15]
        #reshape x to [1024, 12, 15]
        input_seqs = input_seqs.view(input_seqs.size(0), input_seqs.size(1) * 2, input_seqs.size(3))
        # reshape: [1024, 15, 12]  for transformer
        input_seqs = input_seqs.permute(0, 2, 1)
        #############################################################################3

        h_masked = self.transformer(input_seqs)

        # h_masked = self.wt_inter_transformer(input_seqs) #[1024, 15, 72]
        #h_masked = self.W_Transform_without_inverseW(input_seqs)           
        # print("after transformer h_masked shape:", h_masked.shape)
        # print("Shape of h_masked:", h_masked.shape)
        # print("masked_pos shape:", masked_pos.shape)
        # print("masked_pos:", masked_pos)
        # print("Max index in masked_pos:", masked_pos.max().item())

        if self.output_embed:
            return h_masked
        
        if masked_pos is not None:
            print("masked_pos shape:", masked_pos.shape)
            # 检查 masked_pos 的最大值是否在 h_masked 的长度范围内
            # print("Max index in masked_pos:", masked_pos.max().item())
            batch_size, seq_len, feature_dim = h_masked.size()
            # num_masked = 2
            # masked_pos = torch.stack([torch.randperm(seq_len)[:num_masked] for _ in range(batch_size)]).to(input_seqs.device)
            # print("new masked_pos shape:", masked_pos.shape)
           
            # assert masked_pos.max().item() < h_masked.size(1), "Index out of bounds in masked_pos"
            #  # 如果 masked_pos 的值超出范围，可以选择将其裁剪到合法范围内
            # masked_pos = torch.clamp(masked_pos, max=h_masked.size(1) - 1)
           # 在使用 masked_pos 之前，确保其索引不会超过序列长度
            # if masked_pos.max().item() >= h_masked.size(1):
            #     masked_pos = torch.clamp(masked_pos, max=h_masked.size(1) - 1)
            # print("adjust max index masked_pos :", masked_pos.max().item())
            # print("masked_pos:", masked_pos)
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h_masked.size(-1))
            print("masked_pos shape:", masked_pos.shape)
            # print("masked_pos:", masked_pos)
            print("Max index in masked_pos:", masked_pos.max().item())
            h_masked = torch.gather(h_masked, 1, masked_pos)

        # print("masked_pos shape:", masked_pos.shape)  #打印掩码
        # print("masked_pos:", masked_pos)
        # print("Max index in masked_pos:", masked_pos.max().item())    
        # print("before linear h_masked shape:", h_masked.shape)
        h_masked = self.activ(self.linear(h_masked))
        # print("after linear h_masked shape:", h_masked.shape) #最后一维是36    
        # h_masked = self.norm_wt(h_masked)
        h_masked = self.norm(h_masked)
        # print("after norm h_masked shape:", h_masked.shape)
        logits_lm = self.decoder(h_masked)
        # print("logits_lm shape:", logits_lm.shape)

        ##############################准备小波逆变换#####################################
        batch_size, seq_length, feature_dim = logits_lm.shape # [1024, 15, 72]
        num_channels = feature_dim // 2 # 72 // 2 = 36
        logits_lm = logits_lm.permute(0, 2, 1) # [1024, 72, 15]
        logits_lm = logits_lm.view(batch_size, num_channels, 2, seq_length) #[1024, 36, 2, 15]

        # print("logits_lm shape before iwt:", logits_lm.shape)
        #执行小波逆变换
        logits_lm = self.iwt_function(logits_lm) # [1024, 36, 30]
        # print("logits_lm shape after iwt:", logits_lm.shape)
        logits_lm = logits_lm.permute(0, 2, 1)
        #print("logits_lm :", logits_lm)
        # print("logits_lm shape:", logits_lm.shape)    
        ####################################3￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥

        return logits_lm
