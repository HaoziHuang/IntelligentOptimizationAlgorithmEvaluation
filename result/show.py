#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 00:16:33 2019

@author: Ryan
"""

import matplotlib.pyplot as plt
import pandas as pd


file_name_CX = './jbs/jbs_CX_result.csv'
file_name_GTX = './jbs/jbs_GTX_result.csv'
file_name_PMX = './jbs/jbs_PMX_result.csv'

df_CX = pd.read_csv(file_name_CX)
df_GTX = pd.read_csv(file_name_GTX)
df_PMX = pd.read_csv(file_name_PMX)
your_iter = 200

plt.figure(figsize=(16, 9), dpi=40)
plt.plot(df_PMX.values[:your_iter, 2], label='PMX')
plt.plot(df_CX.values[:your_iter, 2], label='CX')
plt.plot(df_GTX.values[:your_iter, 2], label='GTX')
#plt.title('解的距离')
#plt.grid()
plt.legend()
plt.savefig('jbs_解的距离.png')

plt.figure(figsize=(16, 9), dpi=40)
plt.plot(df_PMX.values[:your_iter, 0], label='PMX')
plt.plot(df_CX.values[:your_iter, 0], label='CX')
plt.plot(df_GTX.values[:your_iter, 0], label='GTX')
#plt.title('优化路径长度')
#plt.grid()
plt.legend()
plt.savefig('jbs_优化路径长度.png')
