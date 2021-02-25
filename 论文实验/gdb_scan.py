import datetime
import math
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


# min_pts = 20  # not used in this algorithm


class Point(object):
    d = []
    assigned = True
    index=int
    def __init__(self, d):
        assume(len(d) != 0)
        self.d = d

def cover(p, eps):
    for g in groups:
        if g is not None and distance(g.core, p) <= eps:
            return g
    return None


def cover2(p, eps):

    for g in groups:
        if g is not None and distance(g.core, p) <= (eps * 2):
            return g
    return None

class Group:
    points = []
    index=int

    core = Point

    neighbor_index=[]
    def append(self, p=Point):
        self.points.append(p.index)

    def nearby_groups(self, eps):
        ret = []
        for g in groups:
            if g == self or g is None:
                continue
            if distance(g.core, self.core) <= eps:
                ret.append(g)

        return ret


groups = []


def new_group(x=Point, eps=float):
    g = Group()
    g.core = x
    g.points = find_neighbours(x, eps, g)
    groups.append(g)

    return g


def find_neighbours(p=Point, eps=float, g=Group):

    list_set = set()
    list_set.add(p)
    if distance(p, g.core) <= eps / 2.0:
        for point in g.points:
            list_set.add(point)
    else:
        for point in g.points:
            if distance(p, point) <= eps:
                list_set.add(point)

    nearby_groups = g.nearby_groups(eps)
    for group in nearby_groups:
        if not (distance(group.core, g.core) <= distance(group.core, p) + eps + distance(g.core, p)):
            continue
        for point in group.points:
            if distance(point, p) <= eps:
                set.add(point)

    return list(list_set)





def gdb_scan(points,cluster,this_eps):

    for i in range(len(points)):
        points[i].assigned = True

    for p in points:
        g = cover(p, this_eps)

        if g is not None:
            g.append(p)
            continue

        g2 = cover2(p, this_eps)
        if g2 is not None:
            p.assigned = False
            # continue
        else:new_group(p, eps=this_eps)

    for p in points:
        if p.assigned:
            continue
        g = cover(p, this_eps)
        if g is not None:
            g.append(p)
        else:
            new_group(p, eps=this_eps)
    cluster=getCluster(cluster)
    return cluster


def getPoint(data):
    p=[]
    cluster = [-1 for _ in range(len(data))]  # 聚类

    for i in range(len(data)):
        ippp=Point(data[i])
        ippp.index=i

        p.append(ippp)

    return p,cluster




def getCluster(cluster):
    cluster=cluster
    i = -1

    for g in groups:
        i += 1
        if g is None:
            continue

        cor=g.core
        cluster[cor.index] = i
        for p in g.points:
            if type(p)!=int:
                continue
            cluster[p]=i
    return cluster


# Euler
def distance(p1=Point, p2=Point):

    if len(p1.d) != len(p2.d):
        return 0
    dist = np.sqrt(np.sum(np.square(np.array(p1.d) - np.array(p2.d))))


    return dist


def assume(expression, reason="Expression Is Not True."):
    if expression is None or expression == False:
        if reason is None:
            raise Exception("AssertionError")
        else:
            raise Exception(reason)



def presion(y_true, y_pred):


    acc = accuracy_score(y_true, y_pred)
    return acc


if __name__ == '__main__':


    # # 获取3D8M数据集
    # D8M = pd.read_csv("data/3D0.4M.CSV")
    # data = (D8M).values

    # 获取HIGGS数据集
    HIGGS = pd.read_csv("data/HIGGS1800.csv")

    target = HIGGS['0'].values

    HIGGS = HIGGS.drop(['0'], axis=1)
    data = (HIGGS).values


    # data=data.values

    data,cluster=getPoint(data)

    begin = datetime.datetime.now()
    eps=4.8
    cluster=gdb_scan(data,cluster,eps)
    print(cluster)
    print(eps)
    end = datetime.datetime.now()

    # 得到时间
    totalTime = (end - begin).total_seconds()
    print(set(cluster))

    # score=presion(lable,cluster)

    print("精确度")
    print(accuracy_score(target, cluster))

    print("总时间")
    print(totalTime)





