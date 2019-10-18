#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 01 00:06:33 2019
@author: Ryan

该Python文件存放一些求解TSP问题的GA常用操作
"""
import numpy as np
import os
import random

# =============================================================================
#                             个体随机产生函数
# =============================================================================
def generate_individual(city_n):
    '''
    传入:
        city_n : 城市数量
    返回:
        个体(返回的形状是: (city_n, ))
    注: n(city_n)座城市用0、1、...、n-1来表示
    '''
    individual = np.arange(1, city_n) # 先产生序列
    np.random.shuffle(individual) # 洗牌
    individual = np.concatenate([[0], individual, [0]], axis=0) # 加上收尾城市: 0
    return individual


# =============================================================================
#                             种群随机产生函数
# =============================================================================
def generate_group(individual_quality_of_special, city_n):
    '''
    传入:
        种群个体数 : individual_quality_of_special
        城市数 : city_n
    返回:
        种群 ( 形状是: (individual_quality_of_special, city_n) )
    '''
    group_list = [generate_individual(city_n).reshape(1, -1) for i in range(individual_quality_of_special)]
    return np.concatenate(group_list) # 将 (individual_quality_of_special) 个个体搞在一个种群中


# =============================================================================
#                           计算两地点的欧氏距离
# =============================================================================
def calc_distance_from_coordinate(location_A, location_B):
    '''
    传入:
        location_A, location_B : 两地的坐标
    返回:
        两地的距离
    '''
    x1 = location_A[0]
    y1 = location_A[1]

    x2 = location_B[0]
    y2 = location_B[1]

    return np.sqrt(np.square(x1-x2) + np.square(y1-y2))


# =============================================================================
#                               得到距离关系矩阵
# =============================================================================
def get_relation_array(coordinate_array, save_name):
    '''
    传入:
        coordinate_array : 坐标矩阵
        save_name : 文件名前缀

    返回:
        各城市距离矩阵
    保存距离关系矩阵('./npy_data/' + save_dir + 're_dis_array.npy')
    '''
    # 文件名字(或者已存在的文件名字)
    save_dir = './npy_data/' + save_name + '_re_dis_array.npy'
    try:
        # 如果已存在, 则直接读取
        relation_distance_array = np.load(save_dir)
        return relation_distance_array
    except:
        # 文件不存在则进行正常操作
        pass 
    
    # 城市数量
    city_n = coordinate_array.shape[0]
    # 初始化距离关系矩阵
    relation_distance_array = np.zeros(shape=(city_n, city_n))
    for i_index, i in enumerate(coordinate_array):
        for j_index, j in enumerate(coordinate_array):
            # 给关系矩阵赋值
            relation_distance_array[i_index][j_index] = calc_distance_from_coordinate(i, j)
    # 保存关系矩阵(在当前目录)
    if not os.path.exists('./npy_data'):
        # 如果当前文件夹下
        os.mkdir('./npy_data')
    if not os.path.exists(save_dir):
        # 该文件是否保存, 未存在则保存
        np.save(save_dir, relation_distance_array)
    return relation_distance_array


# =============================================================================
#                      计算个体适应度(以总距离来算)
# =============================================================================
def calc_individual_fitness(individual, relation_distance_array):
    '''
    传入:
        individual : 个体
        relation_distance_array : 关系矩阵
    返回:
        适应度(总距离的相反数)
    '''
    distance = 0
    for j in range(individual.shape[0]-1):
        a = individual[j]
        b = individual[j+1]
        distance += relation_distance_array[a][b]
    # 现在还差最后一个城市和第一个城市的距离
    a = individual[0]
    b = individual[-1]
    distance += relation_distance_array[a][b]
        
    # return -distance                  # 加负号
    return 1 / distance                 # 取倒数


# =============================================================================
#                      计算种群适应度(以总距离来算)
# =============================================================================
def calc_group_fitness(group, relation_distance_array):
    '''
    传入:
        group : 种群
        relation_distance_array : 关系距离矩阵
    返回:
        种群适应度向量
    '''
    fitness_list = []  # 初始化种群适应度的列表
    for individual in group:
        fitness = calc_individual_fitness(individual, relation_distance_array)
        fitness_list.append(fitness)    
    return np.array(fitness_list)   # 设置负号或者倒数移步至函数 calc_individual_fitness


# =============================================================================
#                            部分匹配交叉法
# =============================================================================
def crossover_PMX(individual_A, individual_B):
    '''
    传入:
        individual_A, individual_B : 两条父代染色体
    返回:
        两条子代染色体
    注: 使用 部分匹配交叉 (PMX) 策略
    '''
    city_n = individual_A.shape[0] # 城市总数
    crossover_start = np.random.randint(0, city_n) # 交叉的起始位置 返回 0、...、city_n 的随机数 交叉包括此处
    crossover_length = np.random.randint(1, city_n - crossover_start + 1) # 定义交叉的长度 由于交叉包括终点 则 +1
    
    # 给B用
    B_dict = dict(
                    zip(individual_A[:-1][crossover_start:crossover_start+crossover_length],
                        individual_B[:-1][crossover_start:crossover_start+crossover_length])
                    )
    # 给A用
    A_dict = dict(
                    zip(individual_B[:-1][crossover_start:crossover_start+crossover_length],
                        individual_A[:-1][crossover_start:crossover_start+crossover_length])
                    )

    individual_A_copy = individual_A.copy()       # copy父代
    individual_B_copy = individual_B.copy()       # 即子代的雏形

    # 执行交叉操作
    individual_A_copy[crossover_start:crossover_start+crossover_length] = individual_B[crossover_start:crossover_start+crossover_length]
    individual_B_copy[crossover_start:crossover_start+crossover_length] = individual_A[crossover_start:crossover_start+crossover_length]
    

    '''
    此处要小心  "映射嵌套"
    28 -> 8
    3  -> 26
    30 -> 3
    要将字典转化为
    28 -> 8
    30 -> 26
    以下几段代码就是删除"映射嵌套"
    
    而只改一层嵌套就凉了, 要改彻底, 而边改边检测会报错
    
    RuntimeError: dictionary changed size during iteration
    '''
    while(1):
        '''
        大循环是将映射嵌套都干掉
        '''
        new_A_dict = A_dict.copy()
        new_B_dict = B_dict.copy()
        for key in A_dict.keys():
            if key in A_dict.values():
                new_A_dict[new_B_dict[key]] = new_A_dict[key]
                del new_A_dict[key]
                break # 边改边迭代会报错, 只好break重来
        A_dict = new_A_dict.copy()
        B_dict = {value:key for key,value in A_dict.items()}
        if not set(new_A_dict.keys()) & set(new_A_dict.values()):
            break
    
    
    new_B_dict = {value:key for key,value in new_A_dict.items()}

    for index, i in enumerate(individual_A_copy[:crossover_start]):
        if i in new_A_dict.keys():
            individual_A_copy[index] = new_A_dict[i] # 将冲突的城市更改
       
    
    for index, i in enumerate(individual_B_copy[:crossover_start]):
        if i in new_B_dict.keys():
            individual_B_copy[index] = new_B_dict[i]

            
    for index, i in enumerate(individual_A[crossover_start+crossover_length:-1]):
        if i in new_A_dict.keys():
            individual_A_copy[index+crossover_start+crossover_length] = new_A_dict[i]


    for index, i in enumerate(individual_B[crossover_start+crossover_length:-1]):
        if i in new_B_dict.keys():
            individual_B_copy[index+crossover_start+crossover_length] = new_B_dict[i]
    
    return individual_A_copy, individual_B_copy


# =============================================================================
#                          郭涛交叉(GuoTao Crossover,GTX)
# =============================================================================
def crossover_GTX(individual_A, individual_B):
    '''
    传入:
        individual_A, individual_B : 俩父代个体
    返回:
        一个子代个体
    
    注:使用 郭涛交叉(GuoTao Crossover,GTX) 策略
    '''
    # 城市数量
    city_n = individual_A.shape[0]
    # 从 individual_A 中随机选择一个个体(此处有简化)
    a = np.random.randint(1, city_n+1)
    # 在 individual_B 中找到的此个体的位置
    a_index_in_B = np.where(individual_B == a)[0][0]
    # 在 individual_B 的 b
    try:
        b = individual_B[a_index_in_B + 1]
    except:
        b = individual_B[0]
        
    # 在 individual_A 找到 a, b 的位置
    a_index_in_A = np.where(individual_A == a)[0][0]
    b_index_in_A = np.where(individual_A == b)[0][0]
    
    if abs(a_index_in_A-b_index_in_A) == 1:
        return individual_A
    else:
        big = [a_index_in_A, b_index_in_A][a_index_in_A < b_index_in_A]     
        small = [a_index_in_A, b_index_in_A][a_index_in_A > b_index_in_A]
        # 本来挺帅一 语法糖, 现在有 warning 了
        
        intercept = individual_A[small:(big+1)]                  # +1 越界也无妨
        individual_A_copy = individual_A.copy()                  # 个体 A复制
        individual_A_copy[small:(big+1)] = intercept[::-1]       # 将之间的数字颠倒
        return individual_A_copy


# =============================================================================
#                         循环交叉(Cycle Crossover,CX)    
# =============================================================================
def crossover_CX(individual_A, individual_B):
    '''
    传入:
        individual_A, individual_B : 俩父代个体
    返回:
        俩个子代个体
    
    注: 使用 循环交叉(Cycle Crossover,CX) 策略
    '''
    # 城市数量
    city_n = individual_A.shape[0]
    choose_bool = [False] * city_n
    choose_bool = np.array(choose_bool)
    child_A = individual_A.copy()
    child_B = individual_B.copy()
    choose_int = 0
    while(1):
        if choose_bool[choose_int]:
            # 防止再次进入循环, 无法跳出
            break
        choose_bool[choose_int] = ~choose_bool[choose_int]
        choose_in_B = individual_B[ choose_int ]
        choose_int = np.where(individual_A == choose_in_B)[0][0]
    child_A[~choose_bool] = individual_B[~choose_bool]
    child_B[~choose_bool] = individual_A[~choose_bool]
    return child_A, child_B


# =============================================================================
#                            种群交叉函数
# =============================================================================
def special_CX(special_A, special_B, crossover_func):
    '''
    传入:
        special_A, special_B : 父代种群(两父代种群个体数量应该一致)
        city_n : 城市数量
        crossover_func : 交叉函数
    传出:
        父代子代一家子
    '''
    # 由于 indiv[0]与indiv[-1]位置不动, 所以此处截取
    special_A = special_A[:, 1:-1]
    special_B = special_B[:, 1:-1]
    # 初始化子代容器
    children = []
    if not crossover_func.__name__ == 'crossover_GTX':
        # 其他交叉产生两个子代个体
        for indiv_A, indiv_B in zip(special_A, special_B):
            child_A, child_B = crossover_func(indiv_A, indiv_B)
            children.append(child_A)
            children.append(child_B)
    else:
        # 如果是郭涛交叉, 则只产生一个子代个体
        for indiv_A, indiv_B in zip(special_A, special_B):
            child = crossover_func(indiv_A, indiv_B)
            children.append(child)
    # 产生子代
    children = np.concatenate([children, special_A, special_B], axis=0)
    # 产生全0列
    zero_col = np.zeros(shape=(children.shape[0], 1), dtype=np.int64)
    # 将全零列合并
    return np.concatenate([zero_col, children, zero_col], axis=1)


# =============================================================================
#                                  个体交换变异
# =============================================================================
def mutate_switch_individual(individual):
    '''
    传入:
        未变异的个体, 以及交换子的数量
    返回:
        变异个体
    '''
    individual_copy = individual.copy()
    # 城市数量
    city_n = individual.shape[0]
    # 交换次数, 可以更改交换次数
    n = 1
    for i in range(n):
        a, b = np.random.randint(0, city_n, size=(2,))
        individual_copy[a], individual_copy[b] = individual_copy[b], individual_copy[a]
    return individual_copy


# =============================================================================
#                                  个体倒位变异
# =============================================================================
def mutate_reverse_individual(individual):
    '''
    传入:
        未变异的个体
    返回:
        变异后的个体
    '''
    # 城市数量
    city_n = individual.shape[0]
    # 找寻初始值
    start, stop = np.random.randint(low=0, high=city_n, size=(2,))
    individual_copy = individual.copy()
    
    # 进行翻转操作
    individual[start:stop] = individual[start:stop][::-1]
    return individual_copy


# =============================================================================
#                                 种群变异函数
# =============================================================================
def special_mutate(special, mutate_func, p_m=0.1):
    '''
    传入:
        special : 未变异种群
        mutate_func : 变异函数
    返回:
        变异种群
    '''
    special_copy = special.copy()
    for index, indiv in enumerate(special_copy):
        # if np.random.uniform() < p_m:
        special_copy[index] = mutate_func(indiv)
    return special_copy
        

# =============================================================================
#                             轮盘赌原则进行选择
# =============================================================================
def choose_roulette(special, special_fitness=None, n=None):
    '''
    传入:
        special : 种群 其适应度 及其子代数量n
        special_fitness : 子代适应度(可以不传如)
        n : 子代个数
    返回 选择之后的子代
    
    注: 
        若 special 只是一个种群, 则正常进行没啥问题
        若 special 传入一个种群列表, 则将其合并, 重新计算其整体适应度
    
    (咱也多态一把, 要是没传适应度, 咱就自己算一把)
    '''
    if type(special) == type([]): # 此处不够优雅
        merge_special = np.concatenate(special)
        merge_special_fitness = calc_group_fitness(merge_special)
        new_special = random.choices(population = merge_special, 
                                     weights = merge_special_fitness/np.sum(merge_special_fitness), 
                                     k=n)
    else:
        if type(special_fitness) == type(None):
            special_fitness = calc_group_fitness(special)
        new_special = random.choices(population = special, 
                                     weights = special_fitness/np.sum(special_fitness), 
                                     k=n) # 不得不说这个 random.choices 还是省了不少事情的
        
        # 注：此 random.choices 函数 k != 1 时, 返回的是列表
    new_special = np.array(new_special)
    return new_special[:n//2], new_special[n//2:]


# =============================================================================
#                           锦标赛原则进行选择
# =============================================================================
def choose_championship(special, special_fitness=None, n=None):
    '''
    传入:
        special : 种群 其适应度 及其子代数量n
        special_fitness : 子代适应度(可以不传如)
        n : 子代个数
    返回 选择之后的子代
    
    注: 
        若 special 只是一个种群, 则正常进行没啥问题
        若 special 传入一个种群列表, 则将其合并, 重新计算其整体适应度
    
    (咱也多态一把, 要是没传适应度, 咱就自己算一把)
    '''
    if type(special) == type([]): # 此处不够优雅
        special = np.concatenate(special)
        special_fitness = calc_group_fitness(special)
    
    # 获奖队员
    champions = []
    # 锦标赛全体对象
    all_candidate = np.arange(special.shape[0]).tolist()
    for i in range(n):
        # 洗牌
        np.random.shuffle(all_candidate)
        # 选择锦标赛对象
        a = all_candidate.pop()
        b = all_candidate.pop()
        if special_fitness[a] >= special_fitness[b]:
            # 败者回家
            all_candidate.append(b)
            
            champions.append(a)
        else:
            # 败者回家
            all_candidate.append(a)
            # 胜者为王
            champions.append(b)
    # 获奖队员名单打印
    champions = np.array(champions)
    # 给获奖队员颁奖(子代赋值)
    child = special[champions]
    return child[:n//2], child[n//2:] 
      
    
def choose_best(special, special_fitness, n):
    '''
    此是最佳个体保留？
    传入的参数必须是 (x, self.city_n)
    最佳个体保留选择, 无需传入适应度, 此处重新计算
    
    返回 子代
    '''

    
#    merge_special = np.concatenate(arg)                                     # 将所有种群合并
#    merge_special_fitness = calc_group_fitness(merge_special)               # 计算合并种群的 fitness
#
    special_fitness_order  = special_fitness.argsort()[::-1]    # 将其 index 排序, 这是从小到大排序, 故 [::-1]
#    
    special = special[special_fitness_order]                    # 重新给种群和适应度排序
    special_fitness = special_fitness[special_fitness_order]
    
    child = special[:n]          # 将排在前面提出来作为子代
    
    np.random.shuffle(child)  # 洗牌, 此函数是在原地操作
    
    child_A = child[:n//2]   # 一半一半
    child_B = child[n//2:]
    return child_A, child_B
    

####################################################################################
############################## 引入交换子和交换序 ##################################
####################################################################################
def get_diff_of_2indiv(self, individual_A, individual_B):
    '''
    此函数即求解两智能体的差
    
    indiv_A - indiv_B 的意思是 indiv_B 想变成 indiv_A要经过多少次变换 -> B向A看齐
    '''
    indiv_A = individual_A.copy()
    indiv_B = individual_B.copy() # 先copy
    diff_list = [] # 交换子列表 —— 论文中的 steps
    for indiv_B_i in range(self.city_n-1): # 一包茶一根烟，一个BUG写一天......
        '''
        取 indiv_B[indiv_B_i] 在 indiv_A上滑动
        直到 indiv_A[local_A] == indiv_B[indiv_B_i]
        此时交换indiv_B中 indiv_B[local_A] 和 indiv_B[indiv_B_i]
        同时将元祖 (indiv_B_i, local_A) 添加到 diff_list
        '''
        
        if indiv_A[indiv_B_i] == indiv_B[indiv_B_i]: # 相等就不换了
            continue
        local_A = np.argwhere(indiv_B == indiv_A[indiv_B_i])[0][0]
        
        diff_list.append((indiv_B_i, local_A)) # 添加元组 (sen_B_i, local_A) 到 diff_list
        
        # 交换B中 indiv_B[local_A] 与 indiv_B[indiv_B_i]
        # 最蠢的中间变量交换法
        # temp_int = indiv_B[local_A]
        # indiv_B[local_A] = indiv_B[indiv_B_i]
        # indiv_B[indiv_B_i] = temp_int
        indiv_B[local_A], indiv_B[indiv_B_i] = indiv_B[indiv_B_i], indiv_B[local_A]
    return diff_list

def add_indiv_steps(indiv, steps):
    '''
    智能体与 steps 求和函数
    '''
    indiv_copy = indiv.copy()
    for step in steps:
        temp = indiv_copy[ step[0] ]
        indiv_copy[ step[0] ] = indiv_copy[ step[1] ]
        indiv_copy[ step[1] ] = temp
        # indiv_copy[ step[0] ], indiv_copy[ step[1] ] = indiv_copy[ step[1] ], indiv_copy[ step[0] ]
        # 上面是什么 BUG ??? 这不是 Python的特性吗？？ 以后改
    return indiv_copy

def mul_num_steps(num, steps):
    '''
    数乘 steps 的函数
    由于未解决 steps 是 np.array 的问题
    此处 steps 是 list (应该所有的都是list)
    后续会改
    '''

    if num >= 1:
        # steps_temp = np.concatenate([steps] * int(num))
        steps_temp = [steps] * int(num)
    else:
        assert num>=0, "传入的乘数不能为非正数"
    
#        if steps_temp == None:
#            steps_temp = steps[:int(len(steps)*num)]
#            return steps_temp
#        else:
#            steps_temp2 = steps[:int(len(steps)*(num-int(num)))]
#            return np.concatenate((steps_temp, steps_temp2))
    
    try:
        steps_temp2 = steps[:int(len(steps)*(num-int(num)))]
        # return np.concatenate((steps_temp, steps_temp2))
        return steps_temp + steps_temp2
    
    except NameError:
        steps_temp = steps[:int(len(steps)*num)]
        return steps_temp