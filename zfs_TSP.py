#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 00:06:33 2019

@author: Ryan
"""

import numpy as np
a = np.array([4, 0, 2, 3, 1, 4])
b = np.array([0, 2, 3, 4, 1, 0])

# =============================================================================
#                     两序列距离计算函数
# =============================================================================
def get_distance_of_sequen(sequen_A, sequen_B):
    '''
    传入:
    sequen_A: 被减序列
    sequen_B: 减序列
    
    返回:
    distance: 序列距离
    sub_sequen_list: 两序列的相邻子之差
    '''
    
    # 判定序列是否有效
    if not sequen_A[0] == sequen_A[-1]:
        raise ValueError('种群个体序列必须为哈密顿圈')
    if not sequen_B[0] == sequen_B[-1]:
        raise ValueError('种群个体序列必须为哈密顿圈')
    
    if not ((type(sequen_A) == np.ndarray) and (type(sequen_B) == np.ndarray)):
        raise ValueError('种群个体必须是numpy数组')
    
    # 给距离赋初始值
    distance = 0
    # 两个体序列差: 存放不同相邻子的list
    sub_sequen_list = []
    
    # 将B个体序列转为bytes
    seq_B_bytes = sequen_B.tobytes()
    for i in range(len(sequen_A)-1): 
        # 在个体序列A上滑动
        if (not sequen_A[i:i+2].tobytes() in seq_B_bytes) and (not sequen_A[i:i+2][::-1].tobytes() in seq_B_bytes):
            # 检测该子序列是否在B个体序列中, 检测该子序列倒序是否在B个体序列中
            # 不在则距离加一
            distance += 1 
            # 同时添加到 sub_sequen_list 中
            sub_sequen_list.append(sequen_A[i:i+2])
    
    # 返回距离, 返回个体序列相邻子之差
    return distance, sub_sequen_list



def generate_new_sequen(n):
    '''
    产生距离最远的个体序列
    约定:咱们的不动点(第一个点)均为0
    
    传入:
        个体数 : n
    传出:
        
    '''
    # 所有元素组成的 np.array 
    all_meta = np.arange(0, n)
    # 构造一个"二维列表",存放所有组合
    com_array_list = []
    for meta in all_meta:
        for mate in all_meta:
            if meta == mate:
                continue
            com_array_list.append([meta, mate])
    
    com_array = np.array(com_array_list)
    np.random.shuffle(com_array) # 原地操作我喜欢
    # 初始化第一个解序列
    jie_list = [0]
    # 初始化解序列的第一项
    location = np.where(com_array[:, 0] == jie_list[-1])[0] # 找到第一个0所在位置
    # 
    location[1] in jie_list:
    jie_list.append()
    return com_array
    
    
    
print(generate_new_sequen(5))
    
    
    
    