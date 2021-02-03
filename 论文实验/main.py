
import HNSWDBSCAN as HNSW_D
import DBsCAN as ORIdbscan
from sklearn.metrics import accuracy_score
from collections import Counter
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn import datasets
import pandas as pd
import datetime
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

def getData():
    # # 获取数据iris
    # iris = datasets.load_iris()
    # data = iris.data[:, :4]  # #表示我们只取特征空间中的4个维度
    # target = iris.target

    # # 获取D31数据集
    # D31=pd.read_table("D31.txt", header=None)
    # data=(D31[[0,1]]).values
    # target=(D31[2]).values

    # # 获取t4.8k数据集
    # D31 = pd.read_csv("t4.8k.csv", header=None)
    # data = (D31[[0, 1]]).values
    # target = (D31[2]).values

    # 获取788数据集
    D31 = pd.read_csv("788points.csv", header=None)
    data = (D31[[0, 1]]).values
    target = list(range(len(D31)))

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


    acc = accuracy_score(y_ture_lable, y_pred)
    return acc



if __name__ == '__main__':
    # # iris 数据集
    # data,target=getData()       #获取数据
    # eps = 0.436
    # min_Pts = 4

    # # D31 数据集
    # data, target = getData()  # 获取数据
    # eps = 0.8
    # min_Pts = 30

    # # t4.8k数据集
    # data, target = getData()  # 获取数据
    # eps = 8.5
    # min_Pts = 15


    # 788 数据集
    data, target = getData()  # 获取数据
    eps =1.5
    min_Pts = 6


    # 优化后的HNSW-DBSCAN

    begin = datetime.datetime.now()

    C = HNSW_D.DBSCAN(data, eps, min_Pts)
    end = datetime.datetime.now()

    # 得到时间
    totalTime = (end - begin).total_seconds()
    print(totalTime)
    pp = presion(target, C)

    print(pp)

    # 画图
    plt.scatter(data[:, 0], data[:, 1], c=C)
    plt.show()
