import random
import numpy as np
import copy
import hnswlib
from collections import Counter
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn import datasets
import datetime
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA


def graphConstruct(data,lable,M):
    dim = len(data[0])
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=len(data), ef_construction=200, M=M)
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





def mergeCore(core,neighbor,cluster):
    exchange=[]
    core_inSame_cluster=[]
    for i in range(len(core)):
        for j in range(len(core) - 1 - i):
            if i==j:
                continue
            d = []
            d.append(i)
            d.append(j)
            d=set(d)
            if d in exchange:
                continue
            exchange.append(d)
            if len(list(set(neighbor[i]).intersection(set(neighbor[j]))))==0:
                continue
            # 两邻居点有交集
            if len(list(set(neighbor[i]).intersection(set(neighbor[j])))) != 0:
                for core_i in core[i]:
                    for core_j in core[j]:
                        jjs = []
                        jjs.append(i)
                        jjs.append(j)
                        jjs = set(jjs)
                        if jjs in core_inSame_cluster:
                            continue
                        dist = np.sqrt(np.sum(np.square(data[core_i] - data[core_j])))
                        if dist<eps:
                            core_inSame_cluster.append(jjs)

    for i in core_inSame_cluster:
        i=list(i)
        neighbor_1=i[0]
        neighbor_2=i[1]
        n_1_center=neighbor[neighbor_1][0]
        n_1_cluster_label=cluster[n_1_center]
        for c in range(len(cluster)):

            if c not in neighbor[neighbor_2]:
                continue
            cluster[c]=n_1_cluster_label
    return cluster

def filterBorder(core,border,cluster):
    c=[]
    for i in core:
        c=c+i
    c=list(set(c))

    accept=[]
    #遍历噪点的邻居，查看其邻居是否有核心点。
    for border_i in border:
        for j in border_i:
            border_i_center=border_i[0]
            if border_i_center in accept:
                continue
            if j in c:

                accept.append(border_i_center)
                cluster[border_i_center]=cluster[j]
    return cluster



def hnswlibTok(data,eps,min_Pts):                  #使用HNSW查找每个数据点的最近邻

    data_lables=range(len(data))
    p=graphConstruct(data,data_lables,30)

    #-1 是噪点
    cluster = [-1 for _ in range(len(data))]  # 聚类
    cluster_lable=-1


    core=[]                    #核心点集合
    border=[]
    neighbor=[]

    origDatalabel=list(range(len(data)))
    while len(origDatalabel)!=0:
        center=origDatalabel[0]
        origDatalabel.remove(center)
        labels, distance = p.knn_query(data[center], k=len(data))  # len(X)
        labels = labels[0]

        innerNeighbor, outerNeighbor, neighborPoint, reachAblePoint = findNeighbor(labels)

        # 该点是核心点的情况
        if len(neighborPoint) >= min_Pts:  # 领域内点的个数大于阈值，为核心点
            cluster_lable = cluster_lable + 1  # 簇标记加一
            cluster[center] = cluster_lable

            # 外环点向外扩张，查询其密度可达的邻居点。根据其论文解释，只需要查询三个半径内的点。

            reachData = []
            for r in reachAblePoint:
                reachData.append(data[r])

            outerNeighborData = []
            for o in outerNeighbor:
                outerNeighborData.append(data[o])

            reachP = graphConstruct(reachData, reachAblePoint, 20)

            reach_labels, reach_distance = reachP.knn_query(outerNeighborData, k=len(reachData))

            reach_core, reach_core_Neighbor = findReachNeighbor(reach_labels)
            neighborPoint = neighborPoint + reach_core_Neighbor
            neighborPoint.insert(0,center)

            core_p=[]
            core_p.append(center)
            core_p=core_p+reach_core+innerNeighbor
            core.append(core_p)


            # 给这个点
            for n in neighborPoint:
                cluster[n] = cluster_lable

            neighbor.append(neighborPoint)

        # 该点为非核心点的情况
        if len(neighborPoint) < min_Pts:  # 领域内点的个数小于于阈值，为非核心点

            if len(neighborPoint)!=0:       #把噪点加入噪点集合，排查该噪点内是否有核心点。
                neighborPoint.insert(0,center)
                border.append(neighborPoint)
        origDatalabel = list(set(origDatalabel) - set(core_p))

    new_cluster=mergeCore(core,neighbor,cluster)
    print(new_cluster)
    Final_cluster=filterBorder(core,border,new_cluster)
    return new_cluster


def DBSCAN(X, eps, min_Pts):
    k = -1          #初始化聚类簇数 k=-1

    neighbor_list = []  # 用来保存每个数据的邻域

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




def getData():
    # 获取数据iris
    iris = datasets.load_iris()
    data = iris.data[:, :4]  # #表示我们只取特征空间中的4个维度
    target = iris.target
    return data,target
def presion(y_true, y_pred,length):
    y_true_unique=list(set(y_true))         #返回含有多少的分类

    sum = 0
    for t in y_true_unique:
        target_group=list(np.where(y_true == y_true_unique[t])[0])
        cluster = [y_pred[i] for i
                   in target_group]  # 取出分到相同组的index

        cluster_d=[]
        for i in cluster:               #移除噪点，噪点的lable是-1
            if i !=-1:
                cluster_d.append(i)
        if len(cluster_d)>0:

            c=Counter(cluster_d)
            lable_modst=(c.most_common(1)[0][0])

            for j in cluster_d:
                if j==lable_modst:
                    sum=sum+1
    presionz=sum/length
    return presionz


if __name__ == '__main__':

    data,target=getData()       #获取数据
    eps = 0.4
    min_Pts = 9


    #优化后的HNSW-DBSCAN

    begin = datetime.datetime.now()

    C=hnswlibTok(data, eps, min_Pts)
    end = datetime.datetime.now()

    pp=presion(target, C,len(C))
    # print(pp)


    #得到时间
    totalTime=(end-begin).total_seconds()
    print(totalTime)



    # # 画图
    # plt.scatter(data[:, 0], data[:, 1])
    # plt.show()
