import HNSWDBSCAN as HNSW
import DBsCAN as ORIdbscan
from collections import Counter
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn import datasets
import datetime
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

def getData():
    # # 获取数据iris
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
    eps = 0.8
    min_Pts = 15


    #优化后的HNSW-DBSCAN
    print('HNSDBSCAN  的结果')
    begin = datetime.datetime.now()
    #得到预测值
    label_pred = HNSW.DBSCAN(data, eps, min_Pts)
    end = datetime.datetime.now()
    #得到时间
    totalTime=(end-begin).total_seconds()
    print(totalTime)
    presionss1 = presion(target, label_pred, len(data))
    print(presionss1)

    #
    #
    # # 画图
    # plt.scatter(data[:, 0], data[:, 1], c=label_pred)
    # plt.show()
