#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 19:34:59 2019
@author: Ryan

该Python文件存放针对本篇论文的一些函数
"""
import numpy as np

# =============================================================================
#                               得到初始数据函数
# =============================================================================
def getTspArray(TSP_data_file):
    '''
    传入:
        TSP数据文件
    返回:
        TSP数据 array
        
    注 : 针对 http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/att532.tsp 文件所写读入数据预处理函数
    '''
    with open(TSP_data_file, 'r') as f:
        # 读取文件
        raw_list = f.read().split('\n')
        # 删去没用的行
        raw_list = raw_list[6:-1]
        
    # 将字符串转换为数字
    data_array_list = [] # 初始化data_array的list
    for meta in raw_list:
        # 以空格分割为列表
        row = meta.split(' ')[1:]
        # 将行添加到 data_array_list
        data_array_list.append(row)
    # 得到数据 array
    data_array = np.array(data_array_list).astype(int)
    return data_array


# =============================================================================
#                               求个体距离函数
# =============================================================================
def getSequenDistance(sequen_A, sequen_B):
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


# =============================================================================
#                               求种群总距离函数
# =============================================================================
def getAllDistance(special):
    '''
    传入:
        special : 种群
    传出:
        全部距离之和
        全部距离矩阵
    '''
    # 存放所有元素的列表
    all_meta = []
    for index_A, indiv_A in enumerate(special):
        # 存放每一行
        row = []
        for index_B, indiv_B in enumerate(special):
            if index_A == index_B:
                distance = 0
            else:
                distance, _ = getSequenDistance(indiv_A, indiv_B)
            row.append(distance)
        all_meta.append(row)
    all_meta = np.array(all_meta)
    return np.sum(all_meta), all_meta