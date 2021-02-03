from sklearn.datasets import make_blobs

from sklearn.cluster import DBSCAN

import matplotlib; matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

import numpy as np

#利用生成器生成具有三个簇的合成数据集，共1000个样本点，为方便作图，特征维度这里设为2

X,t=make_blobs(n_samples=1000,n_features=2,centers=[[1.2,1.5],[2.2,1.1],[1.5,2.8]],cluster_std=[[0.3],[0.2],[0.25]],random_state=2020)

#生成样本点的分布图

fig=plt.figure(figsize=(8,8))

ax=fig.add_subplot(111)

ax.scatter(X[:,0],X[:,1])

plt.show()