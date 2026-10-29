# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.layers.batch_norm import FrozenBatchNorm2d

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY
from mamba_ssm import Mamba
__all__ = ["GeneralizedRCNN", "ProposalNetwork"]
from .layers import conv3x3, conv1x1,subpel_conv3x3
from .gdn import GDN
from einops import rearrange
import numbers
from .BUAA import SIMO
from .TANet_d4_8 import DWT_Attention_Block

#model: our_dfnet


def dwt_init(x):
    B, C, H, W = x.shape
    # 高度若为奇数，下侧补1行零
    pad_h = 1 if H % 2 != 0 else 0
    # 宽度若为奇数，右侧补1列零
    pad_w = 1 if W % 2 != 0 else 0
    # 补零（仅补右侧和下侧，避免影响左侧/上侧有效特征）
    if pad_h > 0 or pad_w > 0:
        x = torch.nn.functional.pad(
            x, 
            (0, pad_w, 0, pad_h),  # (左,右,上,下)：右侧补pad_w，下侧补pad_h
            mode='constant', 
            value=0
        )

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    H_feature=torch.cat((x_HL,x_LH,x_HH),dim=1)

    return x_LL, H_feature


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)
    


class all_Fusion(nn.Module):
    def __init__(self,input_channel1,input_channel2,input_channel3):
        super(all_Fusion, self).__init__()
        # self.downsize = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        # )
        # level1
        self.MPCA1_23 = MPCA(input_channel1=input_channel1, input_channel2=input_channel2)
        self.MPCA1_34 = MPCA(input_channel1=input_channel1, input_channel2=input_channel2)
        self.MPCA1_45 = MPCA(input_channel1=input_channel1, input_channel2=input_channel2)

        # level2
        self.MPCA2_34 = MPCA(input_channel1=input_channel1, input_channel2=input_channel2)
        self.MPCA2_45 = MPCA(input_channel1=input_channel1, input_channel2=input_channel2)

        # level3
        self.MPCA3_45 = MPCA(input_channel1=input_channel1, input_channel2=input_channel2)
        self.conv = nn.Conv2d(input_channel2, input_channel3, kernel_size=1)

    def forward(self,p2,p3,p4,p5):

        # ----------------------------p2,p3 fusion
        f1_3 = self.MPCA1_23(p2,p3)
        f1_4 = self.MPCA1_34(p3, p4)
        f1_5 = self.MPCA1_45(p4, p5)

        f2_4 = self.MPCA2_34(f1_3,f1_4)
        f2_5 = self.MPCA2_45(f1_4, f1_5)

        f3_5=self.MPCA3_45(f2_4,f2_5)
        result=self.conv(f3_5)
        return  result



class MLCB(nn.Module):
    """
    Modern Local Context Block (MLCB)
    结合深度卷积与层缩放 (Layer Scale) 的现代局部信息提取模块
    """
    def __init__(self, dim, kernel_size=7, expansion_ratio=2):
        super().__init__()
        # 动态计算 padding 以保证分辨率不变
        pad = (kernel_size - 1) // 2 
        
        # 1. 大核深度卷积 (提取局部结构)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=pad, groups=dim)
        self.norm = nn.InstanceNorm2d(dim, affine=True)
        
        # 2. 跨通道特征交互 (类似 FFN)
        hidden_dim = int(dim * expansion_ratio)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        
        # 3. [学术升级] Layer Scale：可学习的层缩放因子，提升 512 维深层特征的训练稳定性
        self.gamma = nn.Parameter(1e-6 * torch.ones((dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # 缩放后再与残差相加
        return residual + self.gamma * x

class MPCA(nn.Module):
    def __init__(self, input_channel1=256, input_channel2=256,dim=512):
        super(MPCA, self).__init__()

        self.downsize = nn.Sequential(
            nn.Conv2d(input_channel1, input_channel2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel2),
            nn.ReLU(inplace=True),
        )


        self.mamba = MambaLayer(512)
        self.output_conv = nn.Conv2d(512, 256, 3, 1, 1)
        # self.attention=DSAM(256)

        self.res_blocks =MLCB(512)

        # --- Dynamic Fusion 模块 ---
        # 对应 f_gp: 全局最大池化
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        # 对应 w_mlp1 和 w_mlp2: 多层感知机 (MLP)
        # 这里使用标准的隐层降维再升维的 MLP 结构，比单层 Linear 更符合 "multilayer" 的描述
        hidden_dim = max(dim // 4, 16) 
        self.w_mlp1 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )
        self.w_mlp2 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

        # 对应公式中的参数 w ∈ R^{C x C}
        # 使用 1x1 卷积来实现跨通道的线性投影，完全等价于 C x C 矩阵乘法
        self.w = nn.Conv2d(dim, dim, kernel_size=1, bias=False)


    def forward(self, x1, x2):
        x1= self.downsize(x1)
        x_middle = torch.cat((x1, x2), dim=1)
        B, C, H, W = x_middle.shape
        # result =self.output_conv(fuse)

        F_SSM = self.mamba(x_middle)  
        F_CNN = self.res_blocks(x_middle)  

        # --- 2. 计算动态权重 ---
        # 合并特征并进行全局最大池化
        F_merged = F_SSM + F_CNN  
        F_S = self.global_max_pool(F_merged).flatten(1)  # [B, C]

        # 计算 Fα 和 Fβ (去掉了原代码中错误的 self.w 相乘)
        F_alpha = self.w_mlp1(F_S)  # [B, C]
        F_beta = self.w_mlp2(F_S)   # [B, C]

        # Softmax 计算注意力权重 α 和 β
        exp_F_alpha = torch.exp(F_alpha)
        exp_F_beta = torch.exp(F_beta)
        
        # 加上 1e-6 防止分母为0
        alpha = exp_F_alpha / (exp_F_alpha + exp_F_beta + 1e-6)  # [B, C]
        beta = exp_F_beta / (exp_F_alpha + exp_F_beta + 1e-6)    # [B, C]

        # 扩展维度用于特征图调制
        alpha = alpha.view(B, C, 1, 1)  
        beta = beta.view(B, C, 1, 1)    

        # --- 3. 动态调制与融合输出 ---
        # 对应公式内侧: (α · F_SSM + β · F_CNN)
        fused_features = (alpha * F_SSM) + (beta * F_CNN)

        # 对应外侧公式: y = w(...)
        # 这里的 w 是 1x1 卷积，满足 w ∈ R^{C x C}
        y = self.w(fused_features)

        result =self.output_conv(y)
    
        # result = self.attention(result)

        
        return result


class MambaLayer(nn.Module):
    """ Mamba layer for state-space sequence modeling

    Args:
        dim (int): Model dimension.
        d_state (int): SSM state expansion factor.
        d_conv (int): Local convolution width.
        expand (int): Block expansion factor.

    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Check model dimension
        assert C == self.dim

        # Convert input from (B, C, H, W) to (B, H*W, C)
        x = x.float().view(B, C, -1).permute(0, 2, 1)

        # Normalize and pass through Mamba layer
        norm_out = self.norm(x)
        mamba_out = self.mamba(norm_out)

        # Convert output from (B, H*W, C) to (B, C, H, W)
        out = mamba_out.permute(0, 2, 1).view(B, C, H, W)

        return out






class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out
class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2):
        super().__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv = conv3x3(out_ch, out_ch)
        self.igdn = GDN(out_ch, inverse=True)
        self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)

    def forward(self, x):
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.igdn(out)
        identity = self.upsample(x)
        out += identity
        return out

class ResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.gdn = GDN(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.skip = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        return out

class Lenhance(nn.Module):
    def __init__(self, in_size, out_size):
        super(Lenhance, self).__init__()
        # self.downsize = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        # )

        self.DWT1 = DWT()
        self.DWT2 = DWT()
        self.DWT3 = DWT()
        self.DWT4 = DWT()


        # self.Lenhance1 = DWT_Attention_Block(in_size, out_size)             
        # self.Lenhance2 = DWT_Attention_Block(in_size, out_size)
        # self.Lenhance3 = DWT_Attention_Block(in_size, out_size) 
        # self.Lenhance4 = DWT_Attention_Block(in_size, out_size) 

        self.Lenhance1 = DWT_Attention_Block(in_size, out_size, fpn_level=2)  # 处理 P2           
        self.Lenhance2 = DWT_Attention_Block(in_size, out_size, fpn_level=3)  # 处理 P3
        self.Lenhance3 = DWT_Attention_Block(in_size, out_size, fpn_level=4)  # 处理 P4
        self.Lenhance4 = DWT_Attention_Block(in_size, out_size, fpn_level=5)  # 处理 P5

    def forward(self, features):
        # input:(p2,p3)
        # #down_sample
        # ----------------------------intra feature

        # wavelet_features_L = {}

        # wavelet_pairs = [
        #     ('p2', self.DWT1),
        #     ('p3', self.DWT2),
        #     ('p4', self.DWT3),
        #     ('p5', self.DWT4),
        # ]

        # 3. 对P2-P5分别执行频域处理（P6不处理，保留原始）

        wavelet_features_p2,H_p2=self.DWT1(features['p2'])
        wavelet_features_p3,H_p3=self.DWT2(features['p3'])
        wavelet_features_p4,H_p4=self.DWT3(features['p4'])
        wavelet_features_p5,H_p5=self.DWT4(features['p5'])
       

        Lehance_p2 = self.Lenhance1(wavelet_features_p2,H_p2)
        Lehance_p3 = self.Lenhance2(wavelet_features_p3,H_p3)
        Lehance_p4 = self.Lenhance3(wavelet_features_p4,H_p4)
        Lehance_p5 = self.Lenhance4(wavelet_features_p5,H_p5)
        # print(Lehance_p2.shape,Lehance_p3.shape,Lehance_p4.shape)
        return Lehance_p2,Lehance_p3,Lehance_p4,Lehance_p5


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
class LightweightFFN(nn.Module):
    """
    轻量前馈网络（GLU门控 + 深度卷积）
    用 depthwise conv 替代 DFFN 的 FFT patch 操作
    expansion_ratio=1.5，参数量约为 DFFN 的 43%
    """
    def __init__(self, dim, expansion_ratio=1.5, bias=False):
        super().__init__()
        hidden = int(dim * expansion_ratio)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3,
                                padding=1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class FSAS(nn.Module):
    def __init__(self, dim, bias):
        super(FSAS, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim, LayerNorm_type='WithBias')

        self.patch_size = 8

    def forward(self, x):
        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        # -------------------------- 关键修改：添加 padding 适配 8x8 patch --------------------------
        patch_size = self.patch_size  # 假设 self.patch_size=8（从 FSAS 类的 __init__ 中定义）
        B, C_q, H, W = q.shape  # 获取 q 的维度（k 与 q 维度一致，无需重复获取）
        
        # 计算需要补充的 padding：让 H/W 变成 patch_size 的整数倍（不足则补，已满足则补 0）
        pad_h = (patch_size - H % patch_size) % patch_size  # 如 H=26 → pad_h=6（26+6=32）
        pad_w = (patch_size - W % patch_size) % patch_size  # 如 W=40 → pad_w=0（已满足）
        
        # 对 q 和 k 同时填充（v 不参与 patch 拆分，无需填充）
        # mode='reflect' 避免 0 填充引入噪声，更适合视觉特征；也可保留 mode='constant'
        q = F.pad(q, (0, pad_w, 0, pad_h), mode='reflect')  # pad 顺序：左、右、上、下
        k = F.pad(k, (0, pad_w, 0, pad_h), mode='reflect')
        # -----------------------------------------------------------------------------------------

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)


        # -------------------------- 关键修改：裁剪回原始尺寸（与 v 匹配） --------------------------
        out = out[:, :, :H, :W]  # 裁剪掉补充的 padding，恢复为 [B, C_q, H, W]（和 v 维度一致）
        # -----------------------------------------------------------------------------------------
        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output

class TransformerBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', att=True):
        super(TransformerBlock, self).__init__()

        self.att = att
        if self.att:
            self.norm1 = LayerNorm(dim, LayerNorm_type)
            self.attn = FSAS(dim, bias)

        # self.norm2 = LayerNorm(dim, LayerNorm_type)
        # self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        if self.att:
            x = x + self.attn(self.norm1(x))

        # x = x + self.ffn(self.norm2(x))

        return x



class MSFR(nn.Module):
    def __init__(self, dim_in, ffn_expansion_factor=2,
                 bias=False):
        super(MSFR, self).__init__()
       
        self.toplayer = nn.Sequential(
                                    nn.Conv2d(dim_in, 256, kernel_size=3, stride=1, padding=1),
                                    TransformerBlock(256, ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) 
        )  # Reduce channels

        # Lateral layers
        self.latlayer1 = nn.Sequential(
                                       ResidualBlockUpsample( dim_in, 256),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
                                       )
        self.latlayer2 = nn.Sequential(
                                       ResidualBlockUpsample(dim_in, 256),
                                       ResidualBlockUpsample(256, 256),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
                                       )
        self.latlayer3 = nn.Sequential(
                                       ResidualBlockUpsample(dim_in, 256),
                                       ResidualBlockUpsample(256, 256),
                                       ResidualBlockUpsample(256, 256),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
                                       )


        self.smooth1 = TransformerBlock(256, ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) 
        self.smooth2 = TransformerBlock(256, ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) 
        self.smooth3 = TransformerBlock(256, ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) 



        self.enhance = TransformerBlock(192, ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) 

        self.p6Decoder = nn.Sequential(nn.MaxPool2d(1, stride=2))

        # self.upsize=ResidualBlockUpsample(dim_in, 256)
    def _upsample_add(self, x, y):
        _, _, H, W = y.size()

        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, input):
       
        # 自上而下，横向连接
        h, w = input.size()[2:]
        input = self.enhance(input)
        f5 = self.toplayer(input)

        p4 = self.latlayer1(input)
        p3 = self.latlayer2(input)
        p2 = self.latlayer3(input)


        f4 = self.smooth1(self._upsample_add(f5, p4))
        f3 = self.smooth2(self._upsample_add(f4, p3))
        f2 = self.smooth3(self._upsample_add(f3, p2))

        f6=self.p6Decoder(f5)
        
        pyramid = {}
        pyramid["p2"] = f2
        pyramid["p3"] = f3
        pyramid["p4"] = f4
        pyramid["p5"] = f5
        pyramid["p6"] = f6
        # The legacy ROI pooler used by this project allocates its index_put_
        # destination as FP32.  Expose FP32 FPN features at this module
        # boundary while keeping AMP enabled inside the reconstruction path.
        return {name: value.float() for name, value in pyramid.items()}

    


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        # --------------------------------------------add start

    
        self.Lenhance = Lenhance(256, 256)
        self.all_fusion_L = all_Fusion(256,256,192)
        self.IWT = IWT()
        self.MSFR = MSFR(192)
        self.net_belle = SIMO(N=192)

       

        for p in self.backbone.parameters():
            p.requires_grad = False
            FrozenBatchNorm2d.convert_frozen_batchnorm(self.backbone)

        for p in self.proposal_generator.parameters():
            p.requires_grad = False
            FrozenBatchNorm2d.convert_frozen_batchnorm(self.proposal_generator)

        for p in self.roi_heads.parameters():
            p.requires_grad = False
            FrozenBatchNorm2d.convert_frozen_batchnorm(self.roi_heads)

        print("trainable params in MSFR:", sum(p.numel() for p in self.MSFR.parameters() if p.requires_grad))
        
        print("total trainable params:", sum(p.numel() for p in self.parameters() if p.requires_grad))


        # --------------------------------------------end
    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None


        
        features = self.backbone(images.tensor)
        Lehance_p2, Lehance_p3, Lehance_p4, Lehance_p5= self.Lenhance(features)
        Fuse_L= self.all_fusion_L(Lehance_p2, Lehance_p3,Lehance_p4, Lehance_p5)

        output = self.net_belle(Fuse_L)
        Fuse_H = torch.zeros_like(output["y_hat"]).repeat(1, 3, 1, 1)
        fuse = self.IWT(torch.cat([output["y_hat"], Fuse_H], dim=1))
        p5_h, p5_w = features["p5"].shape[-2:]
        fuse = fuse[:, :, :p5_h, :p5_w]

        pyramid = self.MSFR(fuse)

       
        batch_size = len(batched_inputs)
        num_pixels=batch_size*batched_inputs[0]["width"]* batched_inputs[0]["height"]  

        bpp_loss = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        bpp_loss = {'bpp_loss': bpp_loss.mean()*0.15}

      
        features=pyramid
        #---------------------------------end

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}

        losses.update(bpp_loss)
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        Lehance_p2, Lehance_p3, Lehance_p4, Lehance_p5 = self.Lenhance(features)
        Fuse_L = self.all_fusion_L(Lehance_p2, Lehance_p3, Lehance_p4, Lehance_p5)

        Fuse_H = torch.zeros_like(Fuse_L).repeat(1, 3, 1, 1)
        fuse = self.IWT(torch.cat([Fuse_L, Fuse_H], dim=1))
        p5_h, p5_w = features["p5"].shape[-2:]
        fuse = fuse[:, :, :p5_h, :p5_w]
        pyramid = self.MSFR(fuse)
        features = pyramid
        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        return results

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    """
    A meta architecture that only predicts object proposals.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
