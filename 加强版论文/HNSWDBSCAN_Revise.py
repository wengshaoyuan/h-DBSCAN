import random
import numpy as np
import copy
import hnswlib
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn import datasets
import datetime
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score

def graphConstruct(data,lable,M):
    dim = len(data[0])
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=len(data),  ef_construction=200, M=20)
    p.add_items(data, lable)
    p.set_ef(50)
    return p
def findNeighbor(labels):


    innerNeighbor = []  # 一半的半径里的点。
    outerNeighbor = []  # 一半的半径外的点。
    neighborPoint = []  # 邻居点
    reachAblePoint = []  # 三个半径内的邻居点，为了查询外环的核心点。

    center = labels[0]
    for j in range(1, len(labels)):  # 返回的是距离最近的k个元素，按顺序输出的。
        dist = np.sqrt(np.sum(np.square(data[center] - data[labels[j]])))

        # 将属于这个点的内核对象，保存起来。认为与核心点是相同类型点
        if dist <= eps / 2:
            innerNeighbor.append(labels[j])


        # 将外环的点保存起来，扩张的时候。以此为基点向外寻找近邻点
        if dist > eps / 2 and dist <= eps:
            outerNeighbor.append(labels[j])

        neighborPoint = innerNeighbor + outerNeighbor

        # 两个半径原则，将三个半径范围内的点加入领域查询。
        if dist < 3 * eps:
            reachAblePoint.append(labels[j])

        if dist > 3 * eps:
            break
    return innerNeighbor,outerNeighbor,neighborPoint,reachAblePoint

def findReachNeighbor(labels):
    reachNeighbor=[]

    core = []
    for i in labels:
        center=i[0]
        neighbor = []
        for j in range(1,len(i)):
            compare_point=i[j]
            dist = np.sqrt(np.sum(np.square(data[center] - data[compare_point])))

            if dist<=eps:
                neighbor.append(compare_point)
            if dist>eps:
                break
        if len(neighbor)>min_Pts:

            core.append(center)
            reachNeighbor=list(set(reachNeighbor+neighbor))
    return core,reachNeighbor









def hnswlibTok(data,eps,min_Pts):                  #使用HNSW查找每个数据点的最近邻

    data_lables=range(len(data))
    p=graphConstruct(data,data_lables,30)
    core_total=[]                    #核心点集合
    border=[]
    neighborCore=[]

    origDatalabel=list(range(len(data)))
    while len(origDatalabel)!=0:
        center=origDatalabel[0]
        origDatalabel.remove(center)        #删除已经访问邻居的点。

        labels, distance = p.knn_query(data[center], k=len(data))  # len(X)
        labels = labels[0]

        innerNeighbor, outerNeighbor, neighbor, reachPoint_label = findNeighbor(labels)
        if len(neighbor)>=min_Pts:
            core_total.append(center)
        neighborCore.append(set(neighbor))




        # if len(neighbor)>=min_Pts:         #在该点为核心点的情况下才需要查询其外环邻居
        #     # 外环点向外扩张，查询其密度可达的邻居点。根据其论文解释，只需要查询三个半径内的点。
        #
        #     old_neighbor=neighbor
        #     reachData = []
        #     for r in reachPoint_label:
        #         reachData.append(data[r])
        #
        #     #存放这一个簇内的核心点
        #     core=[]
        #     core.append(center)
        #     outerNeighborData = []
        #     for o in outerNeighbor:
        #         outerNeighborData.append(data[o])
        #     # reach_p = graphConstruct(reachData, reachPoint_label, 20)
        #     if len(innerNeighbor)+1>=min_Pts:
        #         core=core+innerNeighbor
        #         #从邻居里面把已经认定为是核心点的点删掉。
        #         origDatalabel = list(set(origDatalabel) - set(core))
        #
        #     for i in range(len(core)):
        #         core_total.append(core[i])
        #         neighborCore.append(set(neighbor))
        #
        #
        # if len(neighbor) <min_Pts:  # 边界点筛查，领域内是否有核心点。
        #     #
        #     # if len(neighbor)==0:
        #     #     continue
        #     neighborCore.append(set(neighbor))
        #     neighbor.insert(0,center)
        #     border.append(neighbor)

    core_total=set(core_total)
    return (neighborCore,core_total,border)



def DBSCAN(X, eps, min_Pts):
    k = -1          #初始化聚类簇数 k=-1

    neighbor_list = []  # 用来保存每个数据的邻域

    gama = set([x for x in range(len(X))])  # 初始化未访问样本集合：gama

    cluster = [-1 for _ in range(len(X))]  # 聚类

    neighbor_list,omega_list,border=hnswlibTok(X,eps,min_Pts)



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


    for i in range(len(border)):
        bor_i=set(border[i])
        co=list(set(bor_i).intersection(omega_list))
        if len(co)==0:
            continue
        else:
            core_label=cluster[co[0]]

            bor_center=list(bor_i)[0]

            cluster[bor_center]=core_label

    return cluster


def getData():
    # 获取数据iris
    iris = datasets.load_iris()
    data = iris.data[:, :4]  # #表示我们只取特征空间中的4个维度
    target = iris.target
    return data,target
def presion(y_true, y_pred):

    class_label=list(set(y_true))

    #将相同下标的元素发在一起。
    label_index=[]
    for i in class_label:
        c=[]
        for j in range(len(y_true)):
            if y_true[j]==i:
                c.append(j)
        label_index.append(c)

    # 查看是否正确分类
    y_ture_lable=list(range(len(y_true)))
    for i in label_index:
        pred_label=[]
        for j in i:
            if y_pred[j]==-1:
                continue
            pred_label.append(y_pred[j])


        if len(pred_label)==0:
            max_label=len(class_label)+100
        else:
            max_label = max(pred_label, key=pred_label.count)
        for s in i:
            y_ture_lable[s]=max_label
    print(y_ture_lable)

    print(accuracy_score(y_ture_lable,y_pred))
    return y_ture_lable






if __name__ == '__main__':

    data,target=getData()       #获取数据
    eps = 0.4
    min_Pts = 9


    #优化后的HNSW-DBSCAN

    begin = datetime.datetime.now()

    C=DBSCAN(data, eps, min_Pts)
    end = datetime.datetime.now()



    pp=presion(target,C)

    print(pp)


    #得到时间
    totalTime=(end-begin).total_seconds()
    print(totalTime)



    # # 画图
    # plt.scatter(data[:, 0], data[:, 1])
    # plt.show()
