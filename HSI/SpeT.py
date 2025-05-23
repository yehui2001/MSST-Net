# -*- coding: utf-8 -*-
'''
@Author: Yehui
@Date: 2025-05-19 11:36:56
@LastEditTime: 2025-05-19 14:41:31
@FilePath: /MSST/net.py
@Copyright (c) 2025 by , All Rights Reserved.
'''
import sys
sys.path.append("/yehui/MSST-Net")
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from importlib import reload
import config
reload(config)


# Shallow Feature Extration
class ResBlock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):  
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out


class SFE(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.conv_start1 = nn.Conv2d(args.hsi_channel+args.msi_channel,args.n_features, kernel_size=3, padding=1)
        self.conv_start2 = nn.Conv2d(args.hsi_channel+args.msi_channel, args.n_features, kernel_size=3, padding=1)
        self.conv_end1 = nn.Conv2d(args.n_features, args.n_features, kernel_size=3, padding=1)
        self.conv_end2 = nn.Conv2d(args.n_features, args.n_features, kernel_size=3, padding=1)
        self.RB1 = nn.ModuleList(
            [ResBlock(args.n_features, args.n_features) for _ in range(args.n_resblocks)]
        )
        self.RB2 = nn.ModuleList(
            [ResBlock(args.n_features, args.n_features) for _ in range(args.n_resblocks)]
        )
    def forward(self, y_cat, z_cat):
        y = self.conv_start1(y_cat)
        z = self.conv_start2(z_cat)
        for i in range(self.args.n_resblocks):
            y = self.RB1[i](y)
            z = self.RB2[i](z)
        y = self.conv_end1(y)
        z = self.conv_end2(z)
        return y, z
    
class Attention(nn.Module):
    def __init__(self,n_features,n_heads):
        super().__init__()
        self.n_heads = n_heads # n_heads = M * M
        self.M = int(np.sqrt(n_heads))
        self.n_features = n_features
        self.temprature = n_features ** -0.5
        self.qkv = nn.Conv2d(n_features, n_features * 3, kernel_size=1, bias=False)
        self.project = nn.Conv2d(n_features, n_features, kernel_size=1, bias=False)
    def forward(self, x):
        # x: (b, h, w ,c)
        x = x.permute(0,3,1,2) # (b, c, h, w)
        b,c,h,w = x.shape
        qkv = self.qkv(x) # (b, 3*n_features, h, w)
        q,k,v = qkv.chunk(3, dim=1) # (b, n_features, h, w)
        q = rearrange(q, 'b c (h m1) (w m2) -> b (m1 m2) c (h w)', m1=self.M, m2=self.M)   # (b, n_heads, c, h*w/n_heads)
        k = rearrange(k, 'b c (h m1) (w m2) -> b (m1 m2) c (h w)', m1=self.M, m2=self.M)
        v = rearrange(v, 'b c (h m1) (w m2) -> b (m1 m2) c (h w)', m1=self.M, m2=self.M)
        score = torch.matmul(q, k.transpose(-2, -1))* self.temprature
        weight = score.softmax(dim=-1)
        out = weight @ v # (b, n_heads, c, h*w/n_heads)
        out = rearrange(out, 'b (m1 m2) c (h w) -> b c (h m1) (w m2)',m1=self.M, m2=self.M, h=h//self.M,w=w//self.M) # 
        out = self.project(out) # (b, n_features, h, w)
        out = out.permute(0,2,3,1) # (b, h, w, n_features)
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


'''
class SpeT(nn.Module):
    def __init__(self,n_features, n_heads,n_blocks):
        super().__init__()
        self.SpeT_blocks = nn.ModuleList([ # 该写法错误❌ 无法灵活使用残差连接
            nn.Sequential(
                nn.LayerNorm(n_features),
                Attention(n_features, n_heads),
                nn.LayerNorm(n_features),
                MLP(n_features)
            ) for _ in range(n_blocks)
        ])
'''

class SpeT(nn.Module):
    def __init__(self,n_features, n_heads,n_blocks):
        super().__init__()
        self.SpeT_blocks = nn.ModuleList([])
        for _ in range(n_blocks):
            self.SpeT_blocks.append(nn.ModuleList([
                nn.LayerNorm(n_features),
                Attention(n_features, n_heads),
                nn.LayerNorm(n_features),
                MLP(n_features)
            ]))
    def forward(self,x):
        x = x.permute(0,2,3,1) # (b, h, w ,c)
        print(x.shape)
        for (norm1,attn,norm2,mlp) in self.SpeT_blocks:
            x = attn(norm1(x)) + x # LayerNorm对通道维度归一化
            x = mlp(norm2(x)) + x
        out = x.permute(0,3,1,2).contiguous() #创建一个副本，保证张量在内存中是连续的
        return out
    
if __name__ == '__main__':
    args = config.Args()
    # sfe = SFE(args)
    # y = torch.randn(1, 31, 10, 10)
    # z = torch.randn(1, 3, 80, 80)
    # y_up = F.interpolate(y, size=(80, 80), mode='bilinear', align_corners=False)
    # y_cat = torch.cat((y_up, z), dim=1)
    # z_down = F.interpolate(z, size=(10, 10), mode='bilinear', align_corners=False)
    # z_cat = torch.cat((y, z_down), dim=1)
    # y_out, z_out = sfe(y_cat, z_cat)
    # print(y_out.shape)

    x = torch.randn(1, 64 , 80, 80)
    # D_spe = 32
    # spe_msa = Attention(D_spe,args.n_heads)
    # out = spe_msa(x)
    # print(out.shape)
    SpeT = SpeT(args.n_features, args.n_heads, args.n_blocks)
    out = SpeT(x)   
    print(out.shape)