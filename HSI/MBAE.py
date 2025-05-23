# -*- coding: utf-8 -*-
'''
@Author: Yehui
@Date: 2025-05-23 10:54:28
@LastEditTime: 2025-05-23 13:26:57
@FilePath: /MSST-Net/HSI/MBAE.py
@Copyright (c) 2025 by , All Rights Reserved.
'''
from random import sample
import torch.nn as nn
import torch
import random

class BandEmbed(nn.Module): # 将原始的HSI数据进行嵌入到D_spe维度
    def __init__(self, hsi_channels, embed_dim):
        super().__init__()
        self.embed = nn.Conv2d(hsi_channels, embed_dim,kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.embed(x)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x
    
def ShuffleIndex(index:list, mask_ratio:float):  
    '''
    index表示要选取的波段索引\\
    mask_ratio表示掩盖波段的比例
    '''
    mask_list = []
    if len(index) < 4:
        raise ValueError("The length of index must be greater than 4.")
    if mask_ratio > 1 or mask_ratio < 0:
        raise ValueError("The mask_ratio must be in the range [0, 1].")
    retain_nums = int(len(index) * (1 - mask_ratio)) # 保留的波段数量
    retain_index = index.copy()
    while len(retain_index) > retain_nums:
        # 随机选择一个波段进行掩盖
        random_index = torch.randint(0, len(retain_index), (1,)) # 生成一个波段的索引
        pop_band = retain_index.pop(random_index.item()) # 从剩余波段中去掉波段
        mask_list.append(pop_band) # 将掩盖的波段加入掩盖列表

    sample_list = retain_index.copy() # 表示未掩盖波段的序列
    assert len(mask_list) == int(round(len(index) * mask_ratio)), "mask length must be same as the ratio!!!"
    return mask_list,sample_list

'''By Author 
def ShuffleIndex(index: list, mask_ratio: float):
    mask_list = []
    if len(index) < 4:
        raise ValueError("inputs must be more than 4")
    else:
        remain_length = int(round((1 - mask_ratio) * len(index)))
        retain_index = index.copy()
        while len(retain_index) > remain_length:
            mask = random.choice(retain_index)
            mask_list.append(mask)
            retain_index.remove(mask)

        sample_list = [x for x in index if x not in mask_list]
        assert len(mask_list) == int(round(len(index) * mask_ratio)), "mask length must be same as the ratio!!!"
    return mask_list, sample_list
'''

def MaskEmbed(token_emb, mask_ratio):
    '''
    token_emb: [B, N, C]\\
    mask_ratio: 掩盖比例
    '''
    len = token_emb.shape[1]




if __name__ == "__main__":
    # x = torch.randn(1, 16,224, 224)  # Exam
    # BandEmbed = BandEmbed(16, 64)
    # x = BandEmbed(x)
    # print(x.shape)  # Expected output: torch.Size([1, 64, 224, 224])

    index = [i for i in range(16)]
    mask_ratio = 0.5
    mask_list,sample_list = ShuffleIndex(index, mask_ratio)
    #print(mask_list)
    print(mask_list)
    print(sample_list)