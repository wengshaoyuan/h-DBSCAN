import random
import numpy as np
import copy
import hnswlib

def hnswlibTok(X,eps,min_Pts):                  #使用HNSW查找每个数据点的最近邻
    dim = len(X[0])
    data_lables=range(len(X))
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=len(X), ef_construction=200, M=20)
    p.add_items(X,data_lables)
    p.set_ef(50)
    labels,distance = p.knn_query(X, k=len(X))       #len(X)


    neighbor_list=[]
    omega_list=[]       #核心对象集合
    for i in labels:
        centers=X[i[0]]
        center_neighbor=i
        dist_list=[]
        for j in range(1,len(i)):
            curr=X[i[j]]
            dist = np.sqrt(np.sum(np.square(centers- curr)))
            dist_list.append(dist)

            if dist>eps:                                #找到小于半径的截至索引位置
                center_neighbor=center_neighbor[0:j]
                break
        neighbor_list.append(set(center_neighbor))
        if len(neighbor_list[-1]) >= min_Pts:
            omega_list.append(i[0])  # 将样本加入核心对象集合
    omega_list = set(omega_list)  # 转化为集合便于操作

    return neighbor_list,omega_list


def DBSCAN(X, eps, min_Pts):
    k = -1          #初始化聚类簇数 k=-1

    gama = set([x for x in range(len(X))])  # 初始化未访问样本集合：gama

    cluster = [-1 for _ in range(len(X))]  # 聚类

    neighbor_list,omega_list=hnswlibTok(X,eps,min_Pts)


    while len(omega_list) > 0:
        gama_old = copy.deepcopy(gama)
        j = random.choice(list(omega_list))  # 随机选取一个核心对象
        k = k + 1
        Q = list()
        Q.append(j)
        gama.remove(j)
        while len(Q) > 0:
            q = Q[0]
            Q.remove(q)
            if len(neighbor_list[q]) >= min_Pts:
                delta = neighbor_list[q] & gama
                deltalist = list(delta)
                for i in range(len(delta)):
                    Q.append(deltalist[i])
                    gama = gama - delta
        Ck = gama_old - gama
        Cklist = list(Ck)
        for i in range(len(Ck)):
            cluster[Cklist[i]] = k
        omega_list = omega_list - Ck
    return cluster