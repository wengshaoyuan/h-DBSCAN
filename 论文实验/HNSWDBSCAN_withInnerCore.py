import hnswlib
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import hnswlib
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn import datasets

import datetime

def hnswlibTok(data):                  #使用HNSW查找每个数据点的最近邻
    dim = len(data[0])
    data_lables=range(len(data))
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=len(data), ef_construction=200, M=20)
    p.add_items(data,data_lables)
    p.set_ef(50)
    labels,distance = p.knn_query(data, k=len(data))       #len(X)
    return labels

def findNeighbor(labels,data,eps):
    innerMC=[]
    neighbor = []
    dictss=[]
    centers=data[labels[0]]
    for curr in range(0, len(labels)):
        currData=data[labels[curr]]

        dist = np.sqrt(np.sum(np.square(centers - currData)))
        dictss.append(dist)
        if dist<0.5*eps:
            innerMC.append(labels[curr])
        if dist > eps:  # 找到小于半径的截至索引位置
            neighbor = labels[0:curr]
            break
    return neighbor,innerMC

def asignLable(data,eps,min_Pts):                  #使用HNSW查找每个数据点的最近邻
    rangeQuire=hnswlibTok(data)
    emptyPoinnt=[]
    data_label = list(range(len(data)))
    core = []
    neighbor_dict = {}
    noise=[]

    while len(data_label) != 0:
        center = data_label[0]
        data_label.remove(center)  # 把查询点删除了。
        lable=rangeQuire[center]

        neighbor,innerMC = findNeighbor(lable, data, eps)
        if len(neighbor) >=min_Pts:
            core.append(center)

            # 在内环的邻居数大于密度阈值时，才进行查询。
            if len(innerMC)>=min_Pts:
                core=core+innerMC
                for i in innerMC:
                    if i not in neighbor_dict.keys():
                        neighbor_dict[i]=[]
                data_label = list(set(data_label) - set(innerMC))

        if len(neighbor)==1:    #判断此点是否为空值点
            noise.append(center)
        neighbor_dict[center] = neighbor
    core = set(core)
    noise=set(noise)
    return neighbor_dict,core,noise

def DBSCAN(X, eps, min_Pts):
    k = -1          #初始化聚类簇数 k=-1
    gama = set([x for x in range(len(X))])  # 初始化未访问样本集合：gama
    cluster = [-1 for _ in range(len(X))]  # 聚类
    neighbor_list,omega_list,noise=asignLable(X,eps,min_Pts)


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
            # if q  in list(omega_list):
                delta = set(neighbor_list[q]) & gama
                deltalist = list(delta)
                for i in range(len(delta)):
                    Q.append(deltalist[i])
                    gama = gama - delta
        Ck = gama_old - gama
        Cklist = list(Ck)
        for i in range(len(Ck)):
            cluster[Cklist[i]] = k
        omega_list = omega_list - Ck
    gama=gama-noise
    for i in gama:
        neihbor_noise=neighbor_list[i]
        number = set(neihbor_noise).intersection(omega_list)
        if len(number)==0:
            continue
        if len(number)!=0:
           cluster[i]=cluster[list(number)[0]]
    return cluster
if __name__ == '__main__':
    # # 获取D31数据集
    # D31=pd.read_table("D31.txt", header=None)
    # data=(D31[[0,1]]).values
    # target=(D31[2]).values
    # eps = 0.8
    # min_Pts = 30

    # # 获取house数据集
    # house=pd.read_csv("houser_processed_15000.csv")
    # data=(house).values
    # print(data)
    #
    # eps =10
    # min_Pts = 20

    # # 获取3D8M数据集
    # D8M = pd.read_csv("data/3D0.4M.CSV")
    # data = (D8M).values
    #
    # eps = 0.01
    # min_Pts = 5

    # # 获取HIGGS数据集
    # HIGGS = pd.read_csv("data/HIGGS1800.csv")
    #
    # target = HIGGS['0'].values
    #
    # HIGGS = HIGGS.drop(['0'], axis=1)
    # data = (HIGGS).values

    House=pd.read_csv("data/houser_processed_18000.csv")
    data = (House).values

    # eps = 0.1
    min_Pts = 20
    eps_list = [10, 15, 20, 25, 30]
    for eps in eps_list:
        print(eps)
        # 原始DBSCAN
        begin = datetime.datetime.now()
        C = DBSCAN(data, eps, min_Pts)
        end = datetime.datetime.now()
        # 得到时间
        totalTime = (end - begin).total_seconds()
        print(set(C))
        print("hDbscan")
        print(totalTime)

        print("end")
        print('--------------------------------')
