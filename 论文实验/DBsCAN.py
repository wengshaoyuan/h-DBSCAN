from sklearn import datasets
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import copy
import datetime

def find_neighbor(j, x, eps):
    N = list()
    for i in range(x.shape[0]):
        temp = np.sqrt(np.sum(np.square(x[j] - x[i])))  # 计算欧式距离
        if temp <= eps:
            N.append(i)

    return set(N)


def DBSCAN(X, eps, min_Pts):
    k = -1
    neighbor_list = []  # 用来保存每个数据的邻域
    omega_list = []  # 核心对象集合
    gama = set([x for x in range(len(X))])  # 初始时将所有点标记为未访问
    cluster = [-1 for _ in range(len(X))]  # 聚类


    for i in range(len(X)):
        neighbor_list.append(find_neighbor(i, X, eps))


        if len(neighbor_list[-1]) >= min_Pts:
            omega_list.append(i)  # 将样本加入核心对象集合
    omega_list = set(omega_list)  # 转化为集合便于操作




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
        print(len(gama))

        for i in range(len(Ck)):
            cluster[Cklist[i]] = k
        omega_list = omega_list - Ck


    return cluster



iris = datasets.load_iris()
X = iris.data[:, :4]  # #表示我们只取特征空间中的4个维度

eps = 0.4
min_Pts = 9
begin = datetime.datetime.now()
C = DBSCAN(X, eps, min_Pts)
print(C)
end = datetime.datetime.now()


totalTime=(end-begin).total_seconds()



plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=C)
plt.show()
