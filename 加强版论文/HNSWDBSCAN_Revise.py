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









def hnswlibTok(data,eps,min_Pts):                  #使用HNSW查找每个数据点的最近邻

    data_lables=range(len(data))
    p=graphConstruct(data,data_lables,30)




    core=[]                    #核心点集合
    border=[]
    neighborCore=[]

    origDatalabel=list(range(len(data)))
    while len(origDatalabel)!=0:
        center=origDatalabel[0]
        origDatalabel.remove(center)        #删除已经访问邻居的点。

        labels, distance = p.knn_query(data[center], k=len(data))  # len(X)
        labels = labels[0]

        innerNeighbor, outerNeighbor, neighbor, reachPoint_label = findNeighbor(labels)


        if len(neighbor)>=min_Pts:         #在该点为核心点的情况下才需要查询其外环邻居
            # 外环点向外扩张，查询其密度可达的邻居点。根据其论文解释，只需要查询三个半径内的点。

            reachData = []
            for r in reachPoint_label:
                reachData.append(data[r])

            outerNeighborData = []
            for o in outerNeighbor:
                outerNeighborData.append(data[o])
            # reach_p = graphConstruct(reachData, reachPoint_label, 20)
            reach_labels, reach_distance = p.knn_query(outerNeighborData, k=len(reachData))
            reach_core, reach_core_Neighbor = findReachNeighbor(reach_labels)  # 由外环得到的核心点的邻居点。
            neighbor=neighbor+reach_core_Neighbor

            #这个neighbor是内环加外环的点，已经查询过领域了 不需要重复计算。
            origDatalabel=list(set(origDatalabel)-set(neighbor))


            neighborCore.append(set(neighbor))
            c=reach_core+innerNeighbor
            c.insert(0,center)

            core.append(c)
        if len(neighbor) <min_Pts:  # 边界点筛查，领域内是否有核心点。

            if len(neighbor)==0:
                continue
            neighbor.insert(0,center)
            border.append(neighbor)

    return (neighborCore,core,border)



def DBSCAN(X,eps,min_Pts):
    k = -1          #初始化聚类簇数 k=-1

    gama = set([x for x in range(len(X))])  # 初始化未访问样本集合：gama
    cluster = [-1 for _ in range(len(X))]  # 聚类
    neighbor_list,omega_list,border=hnswlibTok(X,eps,min_Pts)
    core = omega_list

    omega_list=dict(zip(range(len(omega_list)),omega_list))




    while len(omega_list.keys()) > 0:
        gama_old = copy.deepcopy(gama)
        j = random.choice(list(omega_list.keys()))  # 随机选取一个核心对象
        k = k + 1
        gama=gama-set(omega_list[j])      #从集合中删除核心点，表示已经访问了。
        gama=gama-set(neighbor_list[j])


        for i in neighbor_list[j]:          #遍历查看这个核心点的邻居是否有其它的核心点。
            meg_keys=list(omega_list.keys())
            meg_keys.remove(j)

            for t in meg_keys:
                if i not  in omega_list[t]:
                    continue
                if i in omega_list[t]:

                    gama = gama - set(neighbor_list[t])
                    gama=gama-set(omega_list[t])
                    meg_keys.remove(t)
                    omega_list.pop(t)
            #
            #         print('----------')


        Ck=gama_old-gama

        Cklist = list(Ck)
        for i in range(len(Ck)):
            cluster[Cklist[i]] = k

        omega_list.pop(j)  # 已经抽取出的核心对象，从核心点集合进行删除。
    core_c=[]
    for i in core:
        core_c=core_c+i
    core_c=set(core_c)
    for i in range(len(border)):
        bor_i=set(border[i])
        co=list(set(bor_i).intersection(core_c))
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

    C=DBSCAN(data, eps, min_Pts)
    end = datetime.datetime.now()

    pp=presion(target,C,len(C))

    print(pp)


    #得到时间
    totalTime=(end-begin).total_seconds()
    print(totalTime)



    # # 画图
    # plt.scatter(data[:, 0], data[:, 1])
    # plt.show()
