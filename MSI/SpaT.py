# -*- coding: utf-8 -*-
'''
@Author: Yehui
@Date: 2025-05-22 11:18:51
@LastEditTime: 2025-05-22 11:21:26
@FilePath: /MSST-Net/MSI/SpaT.py
@Copyright (c) 2025 by , All Rights Reserved.
'''
import sys
sys.path.append("/yehui/MSST-Net")
import torch
import config
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from importlib import reload


class Attention(nn.Module):
    def __init__(self,n_features,n_heads):
        # D_spa = n_features
        super().__init__()
        self.n_features = n_features
        self.n_heads = n_heads
        self.heads_dim = n_features // n_heads
        self.temprature = self.heads_dim ** -0.5
        self.qkv = nn.Conv2d(self.n_features, self.n_features * 3, kernel_size=1, bias=False)
        self.project = nn.Conv2d(self.n_features, self.n_features, kernel_size=1, bias=False)
    def forward(self, x):
        # x: [B, H, W, C]
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w c -> b c h w') # [B, C, H, W]
        qkv = self.qkv(x) # [B, 3*C, H, W]
        q,k,v = qkv.chunk(3, dim=1) # [B, C, H, W]
        q = rearrange(q, 'b (c m) h w -> b c m (h w)',m=self.n_heads) # 根据通道分多头
        k = rearrange(k, 'b (c m) h w -> b c m (h w)',m=self.n_heads)
        v = rearrange(v, 'b (c m) h w -> b c m (h w)',m=self.n_heads)
        score = torch.matmul(q, k.transpose(-2, -1))* self.temprature
        weight = score.softmax(dim=-1)
        out = weight @ v
        out = rearrange(out, 'b c m (h w) -> b (c m) h w', h=H, w=W)
        out = self.project(out)
        out = rearrange(out, 'b c h w -> b h w c')
        return out

class MLP(nn.Module):
    def __init__(self,n_features, ratio = 4): 
        super().__init__()
        self.conv1 = nn.Conv2d(n_features, n_features*ratio, kernel_size=1)
        self.GELU = nn.GELU()
        self.conv2 = nn.Conv2d(n_features*ratio, n_features, kernel_size=1)
    def forward(self,x):
        # x: (b, h, w ,c)
        x = x.permute(0,3,1,2)
        x = self.conv1(x)
        x = self.GELU(x)
        x = self.conv2(x)
        x = x.permute(0,2,3,1) # (b, h, w ,c)
        return x

class SpaT(nn.Module):
    def __init__(self,n_features, n_heads,n_blocks):
        super().__init__()
        self.SpaT_blocks = nn.ModuleList([])
        for _ in range(n_blocks):
            self.SpaT_blocks.append(nn.ModuleList([
                nn.LayerNorm(n_features),
                Attention(n_features, n_heads),
                nn.LayerNorm(n_features),
                MLP(n_features)
            ]))
    def forward(self,x):
        x = x.permute(0,2,3,1) # (b, h, w ,c)
        for (norm1,attn,norm2,mlp) in self.SpaT_blocks:
            x = attn(norm1(x)) + x # LayerNorm对通道维度归一化
            x = mlp(norm2(x)) + x
        out = x.permute(0,3,1,2).contiguous() #创建一个副本，保证张量在内存中是连续的
        return out

if __name__ == '__main__':
    # Test the Attention module
    args = config.Args()
    x = torch.randn(1, 64 , 80, 80)
    # attention = Attention(C, n_heads)
    # out = attention(x)
    # print(out.shape)  # Expected output shape: [B, H, W, C]

    SpaT = SpaT(args.n_features, args.n_heads, args.n_blocks)
    out = SpaT(x)
    print(out.shape)  # Expected output shape: [B, H, W, C]
