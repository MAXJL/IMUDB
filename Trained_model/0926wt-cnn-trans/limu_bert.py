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

from functools import partial
import pywt
import pywt.data
from wtconv1d import wavelet_transform_1d
from wtconv1d import inverse_wavelet_transform_1d
from wtconv1d import create_wavelet_filter


from wtconv1d import wavelet_transform
from wtconv1d import wavelet_inverse_transform

from wtconv1d import modwt
from wtconv1d import imodwt

from wtconv1d import wavelet_transform_torch
from wtconv1d import inverse_wavelet_transform_torch


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
        self.variance_epsilon = variance_epsilon
    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta
    
class LayerNorm_high(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.hidden_high), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(cfg.hidden_high), requires_grad=True)
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
        if not isinstance(x, torch.Tensor):
            raise TypeError("Expected input to be a torch.Tensor, but got {}".format(type(x)))
        seq_len = x.size(1) #获取序列长度
        #创建一个表示位置的张量，大小为（B, S） B是batch_size，S是序列长度
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len) # (S,) -> (B, S)
        e = self.lin(x)# factorized embedding#接下来通过线性层lin对输入x进行变换，得到嵌入e
        #如果emb_norm为True，则对e进行LayerNorm归一化操作
        if self.emb_norm:
            e = self.norm(e)
        #将位置编码pos加到嵌入e上
        e = e + self.pos_embed(pos)
        # print("Embeddings e shape:", e.shape)
        #再次归一化
        return self.norm(e)
class Embeddings_high(nn.Module):
    def __init__(self, cfg, pos_embed=None):
        super().__init__()
        # factorized embedding   cfg feature映射到hidden 
        self.lin = nn.Linear(cfg.feature_num, cfg.hidden_high)
        if pos_embed is None: #若没有传入pos_embed，则创建一个pos_embed
            self.pos_embed = nn.Embedding(cfg.seq_len, cfg.hidden_high) # position embedding
        else:
            self.pos_embed = pos_embed
        self.norm = LayerNorm_high(cfg) #初始化了一个LayerNorm模块
        self.emb_norm = cfg.emb_norm
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Expected input to be a torch.Tensor, but got {}".format(type(x)))
        seq_len = x.size(1) #获取序列长度
        #创建一个表示位置的张量，大小为（B, S） B是batch_size，S是序列长度
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len) # (S,) -> (B, S)
        e = self.lin(x)# factorized embedding#接下来通过线性层lin对输入x进行变换，得到嵌入e
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
        # self.dynamic_weights = DynamicWeights(cfg) 
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
        # print("MultiHeadedSelfAttention scores shape:", scores.shape)
        # scores = self.dynamic_weights(scores)  # Add dynamic weights
        scores = F.softmax(scores, dim=-1)
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        h = merge_last(h, 2)
        self.scores = scores
        # print("MultiHeadedSelfAttention h shape:", h.shape)
        return h
class MultiHeadedSelfAttention_high(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden_high, cfg.hidden_high)
        self.proj_k = nn.Linear(cfg.hidden_high, cfg.hidden_high)
        self.proj_v = nn.Linear(cfg.hidden_high, cfg.hidden_high)
        self.scores = None # for visualization
        # self.dynamic_weights = DynamicWeights(cfg) 
        self.n_heads = cfg.n_heads_high
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
        # print("MultiHeadedSelfAttention scores shape:", scores.shape)
        # scores = self.dynamic_weights(scores)  # Add dynamic weights
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
    
class PositionWiseFeedForward_high(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_high, cfg.hidden_ff_high)
        self.fc2 = nn.Linear(cfg.hidden_ff_high, cfg.hidden_high)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        # print("PositionWiseFeedForward x shape:", self.fc2(gelu(self.fc1(x))).shape)
        return self.fc2(gelu(self.fc1(x)))



class DynamicWeights(nn.Module): #动态权重 用于动态调整注意力头的权重 
    """ A module to dynamically adjust the weights of attention heads. """

    def __init__(self, cfg):
        super().__init__()
        self.scale_weights = nn.Parameter(torch.ones(cfg.n_heads), requires_grad=True)

    def forward(self, attn_scores):
        # attn_outputs should have shape [batch_size, num_heads, seq_length, hidden_dim]
        # Applying dynamic weights to the outputs of each head
        scaled_scores = attn_scores * self.scale_weights.view(1, -1, 1, 1)

        #查看动态权重
        
        # print("DynamicWeights scale_weights:", self.scale_weights.data)

        return scaled_scores  # Combine heads

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
        # self.norm1 = ContextAwareLayerNorm(cfg)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        # self.norm2 = ContextAwareLayerNorm(cfg)
        self.norm2 = LayerNorm(cfg)

        # self.drop = nn.Dropout(0.4)
    def forward(self, x):
        # print(x.shape
        h = self.embed(x)
        # print("Embedding output h shape:", h.shape)
        for _ in range(self.n_layers):
            h = self.attn(h)
            h = self.norm1(h + self.proj(h))
            # h = self.drop(h)
            h = self.norm2(h + self.pwff(h))
        return h
class Transformer_high(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings_high(cfg)
        # Original BERT not used parameter-sharing strategies
        # self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        # To used parameter-sharing strategies
        self.n_layers = cfg.n_layers_high
        self.attn = MultiHeadedSelfAttention_high(cfg)
        self.proj = nn.Linear(cfg.hidden_high, cfg.hidden_high)
        # self.norm1 = ContextAwareLayerNorm(cfg)
        self.norm1 = LayerNorm_high(cfg)
        self.pwff = PositionWiseFeedForward_high(cfg)
        # self.norm2 = ContextAwareLayerNorm(cfg)
        self.norm2 = LayerNorm_high(cfg)

        # self.drop = nn.Dropout(0.4)
    def forward(self, x):
        # print(x.shape
        h = self.embed(x)
        # print("Embedding output h shape:", h.shape)
        for _ in range(self.n_layers):
            h = self.attn(h)
            h = self.norm1(h + self.proj(h))
            # h = self.drop(h)
            h = self.norm2(h + self.pwff(h))
        return h

class LIMUBertModel4Pretrain(nn.Module):

    def __init__(self, cfg, output_embed=False):
        super().__init__()
        self.transformer = Transformer(cfg) # encoder
        self.fc = nn.Linear(cfg.hidden, cfg.hidden)
        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ = gelu
        self.norm = LayerNorm(cfg)
        self.decoder = nn.Linear(cfg.hidden, cfg.feature_num)
        self.output_embed = output_embed


        self.fc = nn.Linear(cfg.hidden_high, cfg.hidden_high)
        self.linear_high = nn.Linear(cfg.hidden_high, cfg.hidden_high)
        self.norm_high = LayerNorm_high(cfg)
        self.decoder_high = nn.Linear(cfg.hidden_high, cfg.feature_num)

        self.conv1d_low = nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.conv1d_high = nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1)


        # above is  transformer initialization

        # wt_type = 'db1'
        # in_channels = 6
        # self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        # self.wt_function = partial(wavelet_transform_1d, filters = self.wt_filter)
        # self.iwt_function = partial(inverse_wavelet_transform_1d, filters = self.iwt_filter) 



        self.transformer_low = Transformer(cfg) # encoder
        
        self.transformer_high = Transformer_high(cfg) # encoder


    def forward(self, input_seqs, masked_pos=None):
        device = input_seqs.device  # 获取输入张量所在的设备
        input_seqs = input_seqs.to(device)  # 确保输入张量在正确的设备上
        # print("input_seqs shape:", input_seqs.shape)

        ##########################一级小波变换##########################################33
        #使用modwt进行一级小波变换
        wavecoeff = modwt(input_seqs, filters='db1', level=1)
        #分离高频和低频，分别送入transformer
        high_freq_seqs = wavecoeff[:, :, :-6]
        low_freq_seqs = wavecoeff[:, :, -6:]

  
        # Optional:
        low_freq_seqs = self.conv1d_low(low_freq_seqs.permute(0, 2, 1)).permute(0, 2, 1)
        high_freq_seqs = self.conv1d_high(high_freq_seqs.permute(0, 2, 1)).permute(0, 2, 1)

        h_masked_low = self.transformer_low(low_freq_seqs)
        if self.output_embed:
            return h_masked_low 
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h_masked_low.size(-1))
            h_masked_low = torch.gather(h_masked_low, 1, masked_pos)
        h_masked_low = self.activ(self.linear(h_masked_low))
        h_masked_low = self.norm(h_masked_low)
        logits_lm_low = self.decoder(h_masked_low)

        h_masked_high = self.transformer_high(high_freq_seqs)
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h_masked_high.size(-1))
            h_masked_high = torch.gather(h_masked_high, 1, masked_pos)
        h_masked_high = self.activ(self.linear_high(h_masked_high))
        h_masked_high = self.norm_high(h_masked_high)
        logits_lm_high = self.decoder_high(h_masked_high)


        #logit_lm_high和logit_lm_low重新拼接成wavecoeff
        reconstructed_wavecoeff = torch.cat((logits_lm_high, logits_lm_low), dim=2)

        logits_lm = imodwt(reconstructed_wavecoeff, filters='db1', level=1)



        #####################################################################################
        return logits_lm











      #####################二级小波变换############################################
        # input_seqs = input_seqs.permute(0, 2, 1) # 变为[1024, 6, 40]
        # wt_output = self.wt_function(input_seqs) #一级变换 [1024, 6, 2, 20]
        # low_freq_part = wt_output[:, :, 0, :] #低频部分
        # wt_output_level2_low = self.wt_function(low_freq_part) #二级变换 
        # high_freq_part = wt_output[:, :, 1, :] #高频部分
        # wt_output_level2_high = self.wt_function(high_freq_part)
        # #拼接所有二级变换的结果，顺序是第一级低频后的低频，第一级低频后的高频，第一级高频后的低频，第一级高频后的高频
        # concatenated_features = torch.cat([
        #     wt_output_level2_low[:, :, 0, :],    # 第一级低频后的第二级低频
        #     wt_output_level2_low[:, :, 1, :],    # 第一级低频后的第二级高频
        #     wt_output_level2_high[:, :, 0, :],   # 第一级高频后的第二级低频
        #     wt_output_level2_high[:, :, 1, :]    # 第一级高频后的第二级高频
        # ], dim=1)  # 按特征维度拼接
        # input_seqs = concatenated_features.permute(0, 2, 1) 
        # print("input_seqs shape after wavelet transform:", input_seqs.shape)
        ############################################################################
         ##############################二级：小波逆变换##############################################
        # logits_lm = logits_lm.permute(0, 2, 1)
        # #分离优化后的特征
        # second_level_low_low = logits_lm[:, 0:6, :]
        # second_level_low_high = logits_lm[:, 6:12, :]
        # second_level_high_low = logits_lm[:, 12:18, :]
        # second_level_high_high = logits_lm[:, 18:24, :]
        # #重组第二级变换结果以进行逆变换
        # second_level_low_reconstructed = torch.cat([second_level_low_low.unsqueeze(2), second_level_low_high.unsqueeze(2)], dim=2)
        # second_level_high_reconstructed = torch.cat([second_level_high_low.unsqueeze(2), second_level_high_high.unsqueeze(2)], dim=2)
        # #第二级逆小波变换
        # first_level_low_reconstructed = self.iwt_function(second_level_low_reconstructed)
        # first_level_high_reconstructed = self.iwt_function(second_level_high_reconstructed)
        # #重组第一级变换结果以进行最终逆变换
        # first_level_reconstructed = torch.cat([first_level_low_reconstructed.unsqueeze(2), first_level_high_reconstructed.unsqueeze(2)], dim=2)
        # #第一级逆小波变换，重构原始信号
        # original_data_reconstructed = self.iwt_function(first_level_reconstructed)
        # logits_lm = original_data_reconstructed.permute(0, 2, 1)
        # print("logits_lm shape after iwt:", logits_lm.shape)
        ####################################################################################################


        ########################DCT变换############################################
        # input_seqs = input_seqs.permute(0, 2, 1)
        # # print("input_seqs shape before dct:", input_seqs.shape)
        # input_seqs = self.dct_transform(input_seqs)  # shape: [batch, channels, seq_len]
        # # print("input_seqs shape after dct:", input_seqs.shape)
        # input_seqs = input_seqs.permute(0, 2, 1)
        #########################################################################

        #############################DCT逆变换############################################
        # logits_lm = logits_lm.permute(0, 2, 1)
        # logits_lm = self.inverse_dct_transform(logits_lm)  # shape: [batch, channels, seq_len]
        # logits_lm = logits_lm.permute(0, 2, 1)
        #################################################################################