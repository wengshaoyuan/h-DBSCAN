import random
import numpy as np
import copy
import hnswlib
from sklearn.metrics import accuracy_score
from sklearn import datasets
import datetime
def graphConstruct(data):
    dim = len(data[0])
    data_lables = range(len(data))
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=len(data), ef_construction=200, M=20)
    p.add_items(data, data_lables)
    p.set_ef(50)
    return p
def findNeighbor(labels,data):

    for i in labels:
        centers = data[i[0]]
        neighbor = i
        dist_list = []
        for j in range(1, len(i)):
            curr = data[i[j]]
            dist = np.sqrt(np.sum(np.square(centers - curr)))
            dist_list.append(dist)

            if dist > eps:  # 找到小于半径的截至索引位置
                neighbor = neighbor[0:j]
                break
    return neighbor



def hnswlibTok(data,eps,min_Pts):                  #使用HNSW查找每个数据点的最近邻
    p=graphConstruct(data)  #构建遍历层图。

    data_label=list(range(len(data)))

    core=[]
    neighbor_list=[]
    while len(data_label)!=0:
        center=data_label[0]

        lable,distant=p.knn_query(data[center],k=len(data))
        neighbor=findNeighbor(lable, data)


        if len(neighbor)>=min_Pts:
            core.append(center)
        neighbor_list.append(set(neighbor))
        data_label.remove(center)
    core=set(core)
    print(neighbor_list)
    print(core)
    return neighbor_list,core









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
