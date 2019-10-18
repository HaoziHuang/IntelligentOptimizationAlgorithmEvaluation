#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 01:20:21 2019
@author: Ryan Huang
代码开源在我的 github : https://github.com/hackHaozi/Ryan_TSP
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import expit as sigmoid

class some_TSP:
    '''
    所有算法的基类——编程思想*主要*以群体智能算法界的老大哥——遗传算法为准
    包括一些简单的初始化、交叉、变异、选择
    同时, 引入交换子与交换序的概念
    
    代码中的解(即每个智能体)表示方法为**路径表示**
    '''
    
    def __init__(self, data_name=None, max_iter=100, special_quality=1, individual_quality_of_special=100):
        '''
        初始化函数, 供子类调用
        '''
        self.max_iter = max_iter               # 最大迭代次数
        self.counter = 0                       # 模型全局计数器
        self.special_quality = special_quality # 种群数量
        self.individual_quality_of_special = individual_quality_of_special # 每个种群中个体数量
        self.data_dir = data_name              # TSPLIB 中数据的目录地址

        
        # 放置预处理数据的目录
        if os.path.exists('./npy_data'):
            pass
        else:
            os.mkdir('./npy_data')
        
        # 之前是否已经预处理过该文件
        if os.path.exists('./npy_data/' + self.data_dir + '_re_dis_array.npy'):
            self.relation_distance_array = np.load('./npy_data/' + self.data_dir + '_re_dis_array.npy')
            self.city_n = self.relation_distance_array.shape[0]
        else:
            self.data_process()                  # 数据预处理, 给 self.relation_distance_array、self.lun_lat_array 赋值
        print('基类初始化完成')
        
    def __str__(self):

        return '群体智能算法基类'
    
    def __calc_distance_from_latlun(self, A_tuple, B_tuple): # 该方法私有, 通过经纬度算距离的方法就甭改了吧
        '''
        由于 TSPLIB 的数据可能为经纬度, 则需要一个通过经纬度计算距离的函数
        
        根据经纬度计算距离
        A地经纬度 : A_tuple
        B地经纬度 : B_tuple
        
        返回 AB两地的距离S
        一般仅供初始化调用
        经纬度计算距离公式来源：https://segmentfault.com/a/1190000013922206
        '''
        # A_tuple, B_tuple = AB_tuple[:2], AB_tuple[2:]
        
        Lat_A = A_tuple[1] / 180 * np.pi # A省 维度
        Lat_B = B_tuple[1] / 180 * np.pi # B省 维度
        
        Lun_A = A_tuple[0] / 180 * np.pi # A省 经度
        Lun_B = B_tuple[0] / 180 * np.pi # B省 经度
        
        a = Lat_A - Lat_B
        b = Lun_A - Lun_B
        
        S = 2 * np.arcsin( 
                (
                        ( np.sin(a/2) )**2 
                        +
                        np.cos(Lat_A) * np.cos(Lat_B) * np.sin(b/2)**2
                        )**(0.5) 
        ) * 6378.137
        return S
    
    def __calc_distance_from_coordinate(self, location_A, location_B):
        '''
        用于求解两地欧式距离而已
        
        传入两地的坐标, 返回两地的距离
        '''
        x1 = location_A[0]
        y1 = location_A[1]
        
        x2 = location_B[0]
        y2 = location_B[1]
        
        return np.sqrt(np.square(x1-x2) + np.square(y1-y2))
    
    def data_process(self):
        '''
        数据处理函数, 返回各城市距离矩阵
        由于 TSPLIB 的data文件前几行有信息字符串而最后一行有 EOF
        此函数
         - 读入原文本文件 将信息及EOF删除
         - 获得 shape=(self.city_n, 2)的城市经纬度矩阵
         - (注: 城市编号从上到下依次是(0, self.city_n-1))
         - 获得城市之间的距离关系矩阵
         - 保存距离关系矩阵(self.data_dir+'re_dis_array.npy')
         - 将 lun_lat_array 赋值给类属性
         
        本打算将此方法私有, 结果报了:
            AttributeError: 'GA_TSP' object has no attribute '_GA_TSP__data_process'
        不过没必要私有......
        '''
        with open('./TSP_data/'+self.data_dir, 'r') as f:
            file_list = f.readlines()[7:-1]
        self.city_n = len(file_list)                                                       # 给城市数量赋值
        lun_lat_array = np.array([city_str.split()[1:] for city_str in file_list])                # 得到城市经纬度矩阵(字符串版本)
        lun_lat_array = lun_lat_array.astype(dtype=float)                                         # 转换为 float 版本
        self.lun_lat_array = lun_lat_array                                                        # 将该 array 改为类属性, 可视化时要用
        self.relation_distance_array = np.zeros(shape=(self.city_n, self.city_n))   # 初始化距离关系矩阵
        for i_index, i in enumerate(lun_lat_array):
            for j_index, j in enumerate(lun_lat_array):
                self.relation_distance_array[i_index][j_index] = self.__calc_distance_from_latlun(i, j)                      # 给关系矩阵赋值
        np.save('./npy_data/' + self.data_dir + '_re_dis_array.npy', self.relation_distance_array)     # 保存关系矩阵(在当前目录)
    
    

    

        
    def main(self):
        '''
        不多说，这个一定需要继承重写
        '''
        pass

    def data_save(self, indiv, indiv_fitness, someStr=''):
        '''
        运行数据保存函数
        '''
        result_new = np.array([indiv_fitness, indiv])
        save_best_dir = './result/' + self.data_dir.replace('.tsp', '') + '_' + someStr + '_' + 'best.npy'
        self.save_best_dir = save_best_dir
        try:
            result_old = np.load(save_best_dir, allow_pickle=True)
            if result_new[0] > result_old[0]:
                np.save(save_best_dir, result_new)
        except FileNotFoundError:
            np.save(save_best_dir, result_new)
            
    def get_result(self):
        '''
        返回最优值, 同理, 需要继承重写
        '''
        pass
    
    def visualization_latlun(self):
        # 可视化函数
        self.data_process() # 加载要用的数据
        '''
        该Python文件的作用是可视化TSP问题的解
        
        中途遇到点小问题, 已将问题发在我的博客上:
            https://segmentfault.com/a/1190000019476955
        
        同时,代码报了几个warning,在Matplotlib3.3之后将不能在用,所以请注意版本问题
        '''
        
        import os
        import conda
        import numpy as np
        
        conda_file_dir = conda.__file__
        conda_dir = conda_file_dir.split('lib')[0]
        proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
        os.environ["PROJ_LIB"] = proj_lib
        # 以上4句代码用于添加环境变量
        # windows 的话应该不用这么麻烦
        # 我的环境是Ubuntu18.04 从他github上翻出来的
        # 应该是要大改环境变量 我不想纠结了
        
        
        from mpl_toolkits.basemap import Basemap
        import matplotlib.pyplot as plt
        
        best_order = np.load(self.save_best_dir, allow_pickle=True)[1]
        data_array = self.lun_lat_array[best_order]     # 按照给定的顺序重排
        data_array = data_array[:, ::-1]                # 其经纬度反了
        # 建立一个basemap对象
        m = Basemap(projection='mill',
                    llcrnrlat = 30,     # 左下角的纬度
                    llcrnrlon = -70,    # 左下角的经度
                    urcrnrlat = 80,     # 右上角的维度
                    urcrnrlon = 35,     # 右上角的经度
                    resolution='l'      # 分辨率低, 为了加快生成速度
                    )
        # 以下是换算经纬度 得到 lun_lat_array(注意和 self.lun_lat_array 区分)
        lun_lat_list = []
        for lun_lat in data_array:
            lun_lat_list.append(
                    m(*lun_lat)
                    )
        lun_lat_array = np.array(lun_lat_list)
        
        m.drawcoastlines()                            # 画国家分割线
        m.drawcountries(linewidth=1)                  # 设置国家分割线　线宽为２
                  
        
        m.plot(lun_lat_array[:, 0], lun_lat_array[:, 1], 'b*', markersize=5)   # 画点
        # 此处标记看参考 https://matplotlib.org/api/markers_api.html
        
        m.plot(lun_lat_array[:, 0], lun_lat_array[:, 1], color='r', linewidth=1, label='TSP_SA')
        
        fig = plt.gcf()                    # 得到当前的figure     
        # fig.set_size_inches(25, 15)
        
        plt.legend(loc=4)
        plt.title('TSP')
        plt.show()


class SA_TSP(some_TSP):
    '''
    该类仅仅为经典模拟退火算法——两循环
    外面的循环过程调节温度走向
    里面的循环过程需要进行一系列操作来使抽样达到平稳状态
    多初始化"并行"(未完成)
    '''
    def __init__(self, data_name, max_iter=5000, T_max=100, T_min=1e-8, special_quality=1, individual_quality=100):
        '''
        参数说明:
            data_name: ./TSP_data/ 目录下的数据名字, str类型
            max_iter : 最大迭代次数
            special_quality: 种群数量
            individual_quality: "并行"个体数(替代原来的 individual_quality_of_special)
            
            注: special_quality种群数量无需使用, 本类中废弃
                individual_quality_of_special每个种群的个体数无需使用, 本类中废弃(单个体进化)
            T_min, T_max 最大最小温度
        '''
        super().__init__(data_name, max_iter, special_quality, individual_quality)
        self.individual_quality = individual_quality
        self.T_0 = T_max
        # self.T_0_array = np.array([T_max] * self.individual_quality)  
        self.T_K = T_max # 初始化温度
        self.T_min = T_min
    
    def __str__(self):
        # 展示主要参数
        show_str = '数据名称:%s\n' % self.data_dir + \
                   '最大迭代次数:%s\n' % self.max_iter + \
                   '最高温度:%s\n' % self.T_0 + \
                   '最低温度:%s\n' % self.T_min
                   
        return " 模拟退火算法 ".center(36, '#') +'\n'+ show_str + '#'*(36+6)
    
    def calc_individual_fitness(self, indiv):
        '''
        模拟退火算法：
            由于是退火算法, 求解最小值, 故适应度无需加负号
        '''
        return -super().calc_individual_fitness(indiv)
      
    def calc_P(self, delta_f, T_K):
        '''
        Metropolis准则计算接受新解的概率
        '''
        return sigmoid(-delta_f / T_K)
    
    def indiv_evolute(self, operator, T_K, fitness):
        '''
        智能体个体进化函数
        
        进化采用 基类中的 交换变异 
        '''
        new_operator = self.mutate_switch_individual(operator, n=2)
        new_operator_fitness = self.calc_individual_fitness(new_operator)
        delta_f = new_operator_fitness - fitness
        if delta_f <= 0:
            return new_operator, new_operator_fitness
        else:
            if np.random.uniform() < self.calc_P(delta_f, T_K):
                # 按照给定概率接受坏解
                return new_operator, new_operator_fitness
            else:
                return operator, fitness

    ####################################################################################
    ###################################### 降温方式 ####################################
    ####################################################################################
    def log_coolDown(self):
        # 对数降温
        return self.T_0 / np.log10(1 + self.counter)

    def fast_coolDown(self):
        # 快速下降
        return self.T_0 / (1 + self.counter) 
    
    def line_coolDowm(self):
        # 直线下降
        return self.T_0 * (1 - self.counter / self.max_iter)
    
    def expon_coolDown(self, alpha=0.99):
        # 指数降温
        '''
        实验结果下降速度快, 收敛速度也很快
        alpha 要接近1, 但是太接近1也不好, 太不接近1也不好(此即传说中的退火系数)
        '''
        assert 0<=alpha<=1, "alpha值要在(0,1)之间"
        return alpha * self.T_K
    
    
    def main(self):
        
        # 初始化此处智能体——模拟退火算子
        operator = self.generate_individual()
        
        # 计算智能体的fitness(注：为取负号的fitness)
        operator_fitness = self.calc_individual_fitness(operator)
        
        # 若此类要多次运行, 则温度 T_K、counter 要回归初始值, 若只运行一次则忽略即可
        self.T_K = self.T_0
        self.counter = 0
        
        while(self.max_iter > self.counter):
            operator, operator_fitness = self.indiv_evolute(operator, self.T_K, operator_fitness) # 更新算子及其适应度
            self.T_K = self.expon_coolDown()
            if self.T_K <= self.T_min:
                break
            self.counter += 1
            # print(operator_fitness, self.T_K, self.counter)
        self.best_operator, self.best_fitness = operator, operator_fitness # 将最优秀的智能体取出
        self.data_save(self.best_operator, -self.best_fitness, "SA")
        print("运行完毕")

    def get_result(self):
        '''
        用于交互最优值
        '''
        return self.best_operator, round(self.best_fitness, 3)


class GA_TSP(some_TSP):
    '''
    此类实现经典遗传算法
    '''
    def __init__(self, data_name=None, max_iter=10000, special_quality=2, individual_quality_of_special=100, P_c=0.5, P_m=0.05):
        '''
        参数说明:
            data_name: ./TSP_data/ 目录下的数据名字, str类型
            max_iter : 最大迭代次数
            special_quality: 种群数量
            individual_quality_of_special: 每个种群的个体数
            P_c: 交叉概率
            P_m: 变异概率
        '''
        super().__init__(data_name, max_iter, special_quality, individual_quality_of_special)
        self.P_c = P_c
        self.P_m = P_m
    
    def best_record(self, candidate_specials, candidate_specials_fitness):
        '''
        candidate_specials、candidate_specials_fitness可以是列表也可以是np.array
        列表的话注意对应关系
        
        candidate_specials 是 传入的种群(列表)
        candidate_specials_fitness 是传入种群适应度(列表)
        '''

        try:
            if self.best_individual_list == None: # 此处写的不够优雅, 意思是, 如果变量不存在, 则创建
                pass
        except AttributeError:
            # 存储每一代中最优秀的个体
            self.best_individual_list = []
                    
        try:
            if self.best_individual == None:
                pass
        except AttributeError:
            # 用于保存最优秀的个体 self.best_individual [0]为 适应度, [1]为最优个体
            self.best_individual = []
          
        
        try:
            # 一般来说 candidate_specials 应该是列表
            candidate_special = np.concatenate(candidate_specials)
            candidate_special_fitness = np.concatenate(candidate_specials_fitness)
            fitness_order = candidate_special_fitness.argsort()   # 获得从小到大的顺序
            candidate_special = candidate_special[fitness_order]
            candidate_special_fitness = candidate_special_fitness[fitness_order]
        except ValueError:
            # 估计传入的是一个种群
            fitness_order = candidate_specials_fitness.argsort()   # 获得从小到大的顺序
            candidate_special = candidate_specials[fitness_order]
            candidate_special_fitness = candidate_specials_fitness[fitness_order]

        self.best_individual_list.append(candidate_special_fitness[-1])
        try:
            # 记录最优秀的个体
            if self.best_individual[0] < candidate_special_fitness[-1]:
                self.best_individual[0] = candidate_special_fitness[-1]
                self.best_individual[1] = candidate_special[-1]
            else:
                pass
        except:
            self.best_individual.append(candidate_special_fitness[-1])
            self.best_individual.append(candidate_special[-1])
        
    def main(self):
        
        # 初始化种群
        special_A = self.generate_group()
        special_B = self.generate_group()

        
        while(self.max_iter > self.counter):

            # 计算适应度(找寻最优个体)
            special_A_fitness = self.calc_group_fitness(special_A)
            special_B_fitness = self.calc_group_fitness(special_B)
            
            self.best_record(candidate_specials=[special_A, special_A], 
                             candidate_specials_fitness=[special_A_fitness, special_B_fitness])
            
            # 执行交叉操作 (基类中交叉全是个体交叉, 没有整体交叉)
            child_A_list = []
            child_B_list = []
            
            for i in range(self.individual_quality_of_special):
                if np.random.uniform() < self.P_c: 
                    # 满足交叉概率
                    child = self.crossover_CX(special_A[i], special_B[i])
                    child_A_list.append(child[0])
                    child_B_list.append(child[1])      # 将子代添加到子代列表中
                else:
                    child_A_list.append(special_A[i])
                    child_B_list.append(special_B[i])  # 将未变异的父代添加到子代列表中

            # 执行变异操作
            for i, child in enumerate(child_A_list):
                if np.random.uniform() < self.P_m:
                    # 满足变异概率
                    child_A_list[i] = self.mutate_reverse_individual(child)

            for i, child in enumerate(child_B_list):
                if np.random.uniform() < self.P_m:
                    # 满足变异概率
                    child_B_list[i] = self.mutate_reverse_individual(child)          
            
            # 执行选择操作, 先将子代个体变成 np.array
            child_A, child_B = np.array(child_A_list), np.array(child_B_list)
            # special_AB = self.choose_roulette([child_A, child_B, special_A, special_B], n=self.individual_quality_of_special*2)
            # 轮盘赌原则, 在此处适应度函数的情况下不适合
            # special_A = special_AB[:self.individual_quality_of_special]
            # special_B = special_AB[self.individual_quality_of_special:]
            
            special_A, special_B = self.choose_best(child_A, child_B, special_A, special_B)
            self.counter += 1
            print(self.counter)
        self.data_save(self.best_individual[1], -self.best_individual[0], "GA")
    def get_result(self):
        # 用于和用户交互最优解(应该放在基类中)
        return self.best_individual[0]
    

    
    
class PSO_TSP(some_TSP):
    
    def __init__(self, data_name=None, max_iter=500, special_quality=1, individual_quality_of_special=100, w=0.2, c1=None, c2=None):
        super().__init__(data_name, max_iter, special_quality, individual_quality_of_special)
        self.w = w
        self.c1 = np.random.uniform() if c1==None else c1
        self.c2 = np.random.uniform() if c2==None else c2
    
    def best_record(self, specials, specials_fitness):
        '''
        specials 是 传入的种群(列表)
        specials_fitness 是传入种群适应度(列表)
        
        重写基类的 best_record 方法, (目前基类还没有)
        '''
        fitness_order = specials_fitness.argsort()   # 获得从小到大的顺序
        candidate_special = specials[fitness_order]
        candidate_special_fitness = specials_fitness[fitness_order]

        return candidate_special[-1], candidate_special_fitness[-1]

    
    def main(self):
        special = self.generate_group()
        special_fitness = self.calc_group_fitness(special)
        
        # 初始化最佳个体 gbest
        special_gbest = [special[0], special_fitness[0]]
        
        # 初始化 pbest
        special_pbest = special.copy()
        special_pbest_fitness = special_fitness.copy()
        
        # 初始化粒子速度
        speed_list = [[]] * self.individual_quality_of_special
        

        while(self.max_iter > self.counter):
            special_fitness = self.calc_group_fitness(special)
            
            # 记录全局最优
            special_gbest = self.best_record(special, special_fitness)
            
            # 记录 pbest
            for i in range(self.individual_quality_of_special):
                if special_fitness[i] > special_pbest_fitness[i]:
                    special_pbest[i] = special[i]
                    special_pbest_fitness[i] = special_fitness[i]
            
            for index, speed_i in enumerate(speed_list):
                # 分开写 speed 三项的和
                # 更新粒子速度
                speed_1 = self.mul_num_steps(self.w, speed_i)
                speed_2 = self.mul_num_steps(self.c1 * np.random.uniform(),
                                             self.get_diff_of_2indiv(special_pbest[index], special[index]))
                speed_3 = self.mul_num_steps(self.c2 * np.random.uniform(),
                                             self.get_diff_of_2indiv(special_gbest[0], special[index]))
                
                speed = speed_1 + speed_2 + speed_3
                speed_list[index] = speed
                
                # 更新粒子位置
                special[index] = self.add_indiv_steps(special[index], speed)

            self.counter += 1
            print(self.counter)
            print(special_gbest[1])
            
        self.data_save(special_gbest[0], -special_gbest[1], "PSO")
        
class CSO_TSP(some_TSP):
    
    def __init__(self, data_name=None, max_iter=500, special_quality=1, individual_quality_of_special=100, w=0.2, c1=None, c2=None):
        super().__init__(data_name, max_iter, special_quality, individual_quality_of_special)
        self.w = w
        self.c1 = np.random.uniform() if c1==None else c1
        self.c2 = np.random.uniform() if c2==None else c2