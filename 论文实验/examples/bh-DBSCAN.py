#the resulet in dataset of ecoli
import hnswlib
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn import metrics
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
import tracemalloc
import datetime

def purity_score(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)
    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def hnswlibTok(data):     
    dim = len(data[0])
    data_lables=range(len(data))
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=len(data), ef_construction=200, M=20)
    p.add_items(data,data_lables)
    p.set_ef(50)
    labels,distance = p.knn_query(data, k=len(data))   
    
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
        if dist > eps: 
            neighbor = labels[0:curr]
            break
    return neighbor,innerMC

def asignLable(data,eps,min_Pts):  
    rangeQuire=hnswlibTok(data)
    emptyPoinnt=[]
    data_label = list(range(len(data)))
    core = []
    neighbor_dict = {}
    noise=[]

    while len(data_label) != 0:
        center = data_label[0]
        data_label.remove(center) 
        lable=rangeQuire[center]

        neighbor,innerMC = findNeighbor(lable, data, eps)
        if len(neighbor) >=min_Pts:
            core.append(center)

            if len(innerMC)>min_Pts+1:
                core=core+innerMC
                for i in innerMC:
                    if i not in neighbor_dict.keys():
                        neighbor_dict[i]=[]
        if len(neighbor)==0:   
            noise.append(center)
        neighbor_dict[center] = neighbor
    core = set(core)
    noise=set(noise)
    return neighbor_dict,core,noise

def DBSCAN(X, eps, min_Pts):
    k = -1    
    gama = set([x for x in range(len(X))]) 
    cluster = [-1 for _ in range(len(X))]  
    neighbor_list,omega_list,noise=asignLable(X,eps,min_Pts)


    while len(omega_list) > 0:
        gama_old = copy.deepcopy(gama)
        j = random.choice(list(omega_list))
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
    fileSet=["dataSet/datasetWithnoTarget/AGGREGATION.csv",
             "dataSet/datasetWithnoTarget/t4.8k.csv",
             "dataSet/datasetWithnoTarget/D31.csv",
             "dataSet/datasetWithnoTarget/iris.csv"
             ,"dataSet/datasetWithnoTarget/HTRU_2.csv"
            ,"dataSet/datasetWithnoTarget/ecoli.csv"
              ,"dataSet/datasetWithnoTarget/digits.csv"
            ,"dataSet/datasetWithnoTarget/3dSRN3D.csv"
            ,"dataSet/datasetWithnoTarget/household.csv"
            ,"dataSet/datasetWithnoTarget/HIGGS13D.csv"
            ,"dataSet/datasetWithnoTarget/HIGGS28D.csv"]
    
    min_Pts_list=[6,15,30,9,15,30,15,20,40,5,5]
    eps_list=    [1.5,8.5,0.8,0.4,0.3,0.8,8.5,0.1,1,2,2.4]
    for i in range(len(eps_list)):
#         for i in range(len(eps_list)):
        data_withlabels = pd.read_csv(fileSet[i],header=None)
        data = (data_withlabels).values
        df_data = pd.DataFrame(data)
        data=np.array(data)



        currentPeakMemory=[]
        PeakMemory=[]

        begin = datetime.datetime.now()

        C = DBSCAN(data, eps_list[i], min_Pts_list[i])

        end = datetime.datetime.now()

        totalTime = (end - begin).total_seconds()
        print("this is dataset of ",fileSet[i]," the running time is",totalTime," the eps is ",eps_list[i]," the min_pts is ",min_Pts_list[i])

        print("--------------")