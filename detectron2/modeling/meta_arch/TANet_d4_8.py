# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:12:50 2023

@author: xhwch
"""



import torch
import torch.nn as nn


class LKA(nn.Module):
    """
    Scale-Adaptive Large Kernel Attention (SA-LKA)
    支持动态感受野的 LKA，根据 FPN 层级自适应抹除雨雪条纹
    """
    def __init__(self, dim, dilation=3):
        super(LKA, self).__init__()
        # 1. 5x5 深度卷积，保持不变
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        
        # 2. 动态空洞深度卷积
        # 核心公式：padding = (kernel_size - 1) // 2 * dilation = 3 * dilation
        # dilation 越大，感受野越大
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, 
            padding=3 * dilation, 
            groups=dim, 
            dilation=dilation
        )
        # 3. 1x1 卷积
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn
    

class DWT_Attention_Block(nn.Module):
    def __init__(self, in_size, out_size, d=256,fpn_level=2):
        super(DWT_Attention_Block, self).__init__()
        
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        
        # --- 1. 低频特征提取分支 (保持不变) ---
        self.conv_1 = nn.Conv2d(d, d, kernel_size=3, stride=1, padding=1)
        self.relu_1 = nn.LeakyReLU(0.2, True)
        self.conv_4 = nn.Conv2d(d, d, kernel_size=1, stride=1, padding=0)
        self.relu_4 = nn.LeakyReLU(0.2, True)
        
        self.conv_4_1 = nn.Conv2d(d, d, kernel_size=3, stride=1, padding=1)
        self.relu_4_1 = nn.LeakyReLU(0.2, True)
        self.conv_4_2 = nn.Conv2d(d, d, kernel_size=3, stride=1, padding=1)
        self.relu_4_2 = nn.LeakyReLU(0.2, True)
        self.conv_4_3 = nn.Conv2d(d, d, kernel_size=1, stride=1, padding=0)
        self.relu_4_3 = nn.LeakyReLU(0.2, True)

        self.conv_2 = nn.Conv2d(3*d, d, kernel_size=1, stride=1, padding=0)
        self.relu_2 = nn.LeakyReLU(0.2, True)
        
        # --- 2. 低频特征清洗 (防范风险 3：盲人骑瞎马) ---
        # GDA_Block 负责洗净低频特征中的天气域偏移
        self.GDA_Block = GDA_Block(d, d)
        
        # --- 3. 空间掩码生成器 (防范风险 1：掩码退化) ---
        # 接收洗净的低频特征，生成 1 通道的空间注意力权重
        # 引入 InstanceNorm2d 是为了强制网络保持空间响应的方差，防止 Mask 全部输出 0 或 1
        self.spatial_mask_generator = nn.Sequential(
            nn.Conv2d(d, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64), 
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
        # --- 4. 高频特征对齐与提取 ---
        # 复用你的 SKFF，提取多尺度高频边缘
        
        # --- 5. 极度降维瓶颈 (防范风险 2：BPP 严重超载) ---
        # 将过滤后的 256 通道高频，极限压缩到 16 通道，逼迫网络只保留最致命的边缘
        bottleneck_dim = 16 
        self.hf_bottleneck_squeeze = nn.Sequential(
            nn.Conv2d(d, bottleneck_dim, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, True)
        )
        # 恢复到 256 通道以备残差相加，加入 BN 稳定数值分布
        self.hf_bottleneck_expand = nn.Sequential(
            nn.Conv2d(bottleneck_dim, d, kernel_size=1, bias=False),
            nn.BatchNorm2d(d)
        )
        
        # --- 6. 尾部融合 ---
        self.conv_3 = nn.Conv2d(d, d, kernel_size=3, stride=1, padding=1)
        self.relu_3 = nn.LeakyReLU(0.2, True)

        self.LPA_Block = SimAM()
        
        dilation_map = {2: 4, 3: 3, 4: 2, 5: 1}
        lka_dilation = dilation_map.get(fpn_level, 3) # 默认取3
        
        self.GSA_Block = LKA(d, dilation=lka_dilation)

    def forward(self, x_lf, x_hf):
        # x_lf: 低频特征 (B, d, H, W)
        # x_hf: 高频特征组合 (LH, HL, HH, 形状为 B, 3*d, H, W)
        
        # ==========================================
        # 步骤 1：低频特征的提取与彻底清洗
        # ==========================================
        hx = self.conv_1(x_lf)
        hx = self.relu_1(hx)
        hx = self.conv_4(hx)
        hx = self.relu_4(hx)
                
        hx1 = self.conv_4_1(hx)
        hx1 = self.relu_4_1(hx1)
        hx2 = self.conv_4_2(hx)
        hx2 = self.relu_4_2(hx2)
        hx3 = self.conv_4_3(hx)
        hx3 = self.relu_4_3(hx3)
        
        GSA = self.GSA_Block(hx1)
        LPA = self.LPA_Block(hx2)
        
        hx_lf_fused = torch.cat([GSA, LPA, hx3], dim=1)
        hx_lf_fused = self.conv_2(hx_lf_fused)
        hx_lf_fused = self.relu_2(hx_lf_fused)
        hx_lf_fused = hx_lf_fused + x_lf
        
        # 关键点：用 GDA 洗净特征，得到真正的“语义探照灯”源头
        hx_lf_clean = self.GDA_Block(hx_lf_fused)
        
        # ==========================================
        # 步骤 2：生成空间掩码 (Spatial Mask)
        # ==========================================
        # 生成 [B, 1, H, W] 的空间门控，1 代表目标，0 代表背景噪声
        spatial_mask = self.spatial_mask_generator(hx_lf_clean)
        
        # ==========================================
        # 步骤 3：高频特征提取与严格过滤
        # ==========================================
        # hf_list = torch.chunk(x_hf, chunks=3, dim=1)
        # hf_aligned = self.hf_encoder(hf_list) # [B, 256, H, W]

        LH, HL, HH = torch.chunk(x_hf, chunks=3, dim=1)
        # 绝对值相加，完美保留所有方向的边缘能量，0参数，极速计算！
        hf_aligned = torch.abs(LH) + torch.abs(HL) + torch.abs(HH)
        
        # 物理隔绝：利用广播机制，直接把背景区域的高频毛刺乘以 0 抹平
        hf_filtered = hf_aligned * spatial_mask
        
        # ==========================================
        # 步骤 4：极度降维瓶颈 (保 BPP 的核心)
        # ==========================================
        # 256 -> 16 通道，提纯关键边缘
        hf_bottleneck = self.hf_bottleneck_squeeze(hf_filtered)
        # 16 -> 256 通道，准备融合
        hf_ready = self.hf_bottleneck_expand(hf_bottleneck)
        
        # ==========================================
        # 步骤 5：最终融合与输出
        # ==========================================
        hx_out = hx_lf_clean + hf_ready
        
        hx_out = self.conv_3(hx_out)
        hx_out = self.relu_3(hx_out)
        
        hx_out = hx_out + x_lf
        hx_out = hx_out + self.identity(x_lf)
        
        return hx_out

        







class SimAM(nn.Module):
    """
    SimAM: A Simple, Parameter-Free Attention Module
    无参数的 3D 注意力机制，基于能量函数提纯局部细节特征
    """
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        # 计算每个像素与周围像素的方差（能量函数）
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        # 利用 Sigmoid 激活并乘回原特征，实现局部细节锐化
        return x * self.activaton(y)



class GDA_Block(nn.Module):
    """Global Distribution Attention Block"""
    def __init__(self, in_size, out_size):
        super(GDA_Block, self).__init__()
        
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(0.2, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(0.2, inplace=False)
        
        self.norm = nn.InstanceNorm2d(out_size//2, affine=True)


    def forward(self, x):
        out = self.conv_1(x)

        out_1, out_2 = torch.chunk(out, 2, dim=1)
        out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out = out + x

        return out








