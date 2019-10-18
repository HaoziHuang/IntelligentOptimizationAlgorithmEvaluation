#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:14:08 2019

@author: Ryan
"""
import TSP_GA_lib as T_lib
from TSP_temp_lib import getTspArray, getAllDistance

import numpy as np
import pandas as pd

att48_dir = './TSP_data/att48.tsp'
att532_dir = './TSP_data/att532.tsp'

# 选择实验的TSP数据
deal_file = att48_dir # 文件目录
individual_quality = 100 # 种群中的个体数量
max_iter = 2
city_n = int(deal_file.split('/')[-1][3:-4]) # 城市数量

# 获得坐标 array
coor_array = getTspArray(deal_file)

# 求得距离关系矩阵
relationArray = T_lib.get_relation_array(coor_array, deal_file.split('/')[-1])

# 产生两个初始种群
special_A = T_lib.generate_group(individual_quality, city_n)
special_B = T_lib.generate_group(individual_quality, city_n)

## 计算适应度
#special_A_fitness = T_lib.calc_group_fitness(special_A, relationArray)
#special_B_fitness = T_lib.calc_group_fitness(special_B, relationArray)
#
# 
while(max_iter>0):
    # 交叉操作, 得到种群的一大家子
    special_CX = T_lib.special_CX(special_A, special_B, T_lib.crossover_PMX)
    
    # 对这一大家子实行变异操作
    special_mutate = T_lib.special_mutate(special_CX, T_lib.mutate_reverse_individual)
    
    # 计算种群适应度
    special_fitness = T_lib.calc_group_fitness(special_mutate, relationArray)
    
    # 对个体进行选择, 同时A, B分家
    # special_A, special_B = T_lib.choose_best(special_mutate, special_fitness, 2*individual_quality)
    special_A, special_B = T_lib.choose_championship(special_mutate, special_fitness, 2*individual_quality)
    # 种群合体
    special = np.concatenate([special_A, special_B], axis=0)
    result_str = str(1/np.max(special_fitness)) + ', ' + str(max_iter)+', ' + str(getAllDistance(special)[0]) + '\n'
    with open('锦标赛_PMX_result.csv', 'a+') as f:
        f.write(result_str)
    print(result_str, end='')
    max_iter = max_iter - 1