import HNSWDBSCAN_FIN as hdbscan
import DBsCAN as ORIdbscan
import gdb_scan as gdbscan
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
    # # 获取D31数据集
    # D31=pd.read_table("D31.txt", header=None)
    # data=(D31[[0,1]]).values
    # target=(D31[2]).values
    # eps = 0.8
    # min_Pts = 30

    # # 获取house数据集
    # house=pd.read_csv("houser_processed_15000.csv")
    # data=(house).values
    #
    # eps =10
    # min_Pts = 20

    # # 获取3D8M数据集
    # D8M = pd.read_csv("data/3D0.4M.CSV")
    # data = (D8M).values
    #
    # eps = 0.01
    # min_Pts = 5

    # 获取HIGGS数据集
    HIGGS = pd.read_csv("data/HIGGS1000013D.csv")
    data = (HIGGS).values

    eps = 0.1
    min_Pts = 5

    # 优化版HDBSCAN
    begin = datetime.datetime.now()
    C = hdbscan.DBSCAN(data, eps, min_Pts)
    end = datetime.datetime.now()

    print(set(C))

    # 得到时间
    totalTime = (end - begin).total_seconds()
    print("hdbscan")
    print(totalTime)


    # 获取HIGGS数据集
    HIGGS = pd.read_csv("data/HIGGS1000013D.csv")
    data = (HIGGS).values

    eps = 0.1
    min_Pts = 5
    # 原始DBSCAN
    begin = datetime.datetime.now()
    C = ORIdbscan.DBSCAN(data, eps, min_Pts)
    end = datetime.datetime.now()
    # 得到时间
    totalTime = (end - begin).total_seconds()
    print(set(C))
    print("原始dbscan")
    print(totalTime)

    print("end")
