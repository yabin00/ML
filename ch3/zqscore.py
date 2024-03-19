# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 21:09:40 2021

@author: whj
"""
import numpy as np

def ZQ_score(X, labels):
    '''
    计算ZQ系数。
    para X：数组形式的样本点集，每一行是一个样本点。
    para labels：数组形式的测试标签集。
    retrurn: ZQ系数。
    '''
    n_samples = len(X) # 标本总数
    label = list(set(labels)) # 标签列表
    n_labels = len(label) # 标签数
    
    ### 把样本及标签分簇存放
    X_i = []
    y_i = []
    for i in label:
        X_i.append([])
        y_i.append([])
        
    for i in range(n_samples):
        j = label.index(labels[i]) # 该样本在label标签列表中的下标
        X_i[j].append(X[i])
        y_i[j].append(labels[i])
        
    ### 计算簇内众Z距离
    Z_dist = np.zeros(shape = (n_labels)) # 存放每个簇的最近邻距离
    for i in range(n_labels):
        n_cluster = len(X_i[i])
        sample_z_dist = [] # 用来记录簇内每个样本的Z距离
        for j in range(n_cluster):            
            min_dist = np.inf
            for k in range(n_cluster):
                if j == k:
                    continue
                dist = np.sqrt(np.sum(np.square(X_i[i][j] - X_i[i][k]))) # 两个样本间的欧氏距离
                if dist < min_dist:
                    min_dist = dist
            if min_dist == np.inf:
                sample_z_dist.append(0) # 簇内只有一个元素时
            else:
                sample_z_dist.append(min_dist)
        Z_dist[i] = np.mean(sample_z_dist)
   
    ### 计算簇间群Q距离
    Q_dist = np.zeros(shape = (n_labels, n_labels)) # 二维数组，用来存放簇之间的Q距离
    for i in range(n_labels):
        for j in range(n_labels):
            if i == j:
                continue
            i2j_min_dist = [] # 用来记录第i个簇内样本点到第j个簇的最小距离
            for sample1 in X_i[i]:
                min_dist = np.inf
                for sample2 in X_i[j]:
                    dist = np.sqrt(np.sum(np.square(sample1 - sample2))) # 两个样本间的欧氏距离
                    if dist < min_dist:
                        min_dist = dist
                if min_dist < np.inf:
                    i2j_min_dist.append(min_dist)
            Q_dist[i,j] = np.min(i2j_min_dist) # 群距离是样本点之间距离的最小值
            # Q_dist[i,j] = np.min(i2j_min_dist) # 群距离用点到簇的距离来定义
                        
    return np.mean(Z_dist) / ( np.sum(Q_dist) / ( n_labels * (n_labels -1) ) )