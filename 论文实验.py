# -*- coding: utf-8 -*-
import csv
import numpy as np
import pandas as pd

"""
变量说明：day： 当前日期
        finallist ：所有两两关系九十天的相关度字典
        List ：记录一天的路线（key）与编号（value）
        Lines ：二维列表，记录一天中所有线路的刷卡序列
        currday ： 统计天数
        sheet  ：{车牌号：[时间戳,线路,刷卡人数]}

"""
with open(r"E:\所有天数所有线路.csv", 'r') as f:
    reader = csv.reader(f)
    day = '20181001'
    List = {}
    Lines = []
    finallist = {}
    sheet = {}
    currday = 0
    for line in reader:
        if line[0] == day:
            if line[5] not in sheet:  # 记录刷卡信息
                # line[5]车牌号 , line[4]线路
                sheet.setdefault(line[5], []).append(line[1])
                sheet.setdefault(line[5], []).append(line[4])
                sheet.setdefault(line[5], []).append(1)
                # 该车第一次出现
                sheet.setdefault(line[5], []).append(1)
                # 该车最后一次出现的时间戳
                sheet.setdefault(line[5], []).append(line[1])

            # 特别地，第n趟首次刷卡。此时flag=0 && 与上一次车时间戳相隔超过1000
            elif (eval(line[1]) - eval(sheet[line[5]][4])) > 1000:
                sheet[line[5]][4] = line[1]
                # times++;
                sheet[line[5]][3] += 1
                # 修改当前车辆车牌号 = 车牌号 + 出现次数
                line[5] = line[5] + str(sheet[line[5]][3])
                sheet.setdefault(line[5], []).append(line[1])
                sheet.setdefault(line[5], []).append(line[4])
                sheet.setdefault(line[5], []).append(1)
                sheet.setdefault(line[5], []).append(1)
            # 平凡的情况
            elif (line[5] + str(sheet[line[5]][3])) in sheet:
                line[5] = line[5] + str(sheet[line[5]][3])
                sheet[line[5]][2] += 1
            # 统计第一趟车的刷卡人数
            else:
                sheet[line[5]][2] += 1
        else:
            # 此时进入第二天
            day = line[0]
            # 统计同一线路刷卡数据，并汇总成二维列表
            i = 0
            for key in sheet.keys():
                # 路线刷卡数据不存在，写入List
                if sheet[key][1] not in List:
                    Lines.append([])
                    Lines[i].append(sheet[key][2])
                    List.setdefault(sheet[key][1], i)
                    i += 1
                    flag = sheet[key][1]
                    # 两个相邻的同路线刷卡数据合并
                elif sheet[key][1] in List and flag == sheet[key][1]:
                    Lines[List.get(flag)][-1] = Lines[List.get(flag)][-1] + sheet[key][2]
                elif sheet[key][1] in List and flag != sheet[key][1]:
                    Lines[List.get(sheet[key][1])].append(sheet[key][2])
                    flag = sheet[key][1]
            # 将线路与相应刷卡数序列匹配成列表形式
            a = list(zip(List.keys(), Lines))
            for m in range(len(a) - 1):
                for n in range(m + 1, len(a)):
                    # 删除元素使列表长度相等
                    if len(a[m][1]) > len(a[n][1]):
                        L1 = a[m][1][0:len(a[n][1])]
                        L2 = a[n][1]
                    else:
                        L2 = a[n][1][0:len(a[m][1])]
                        L1 = a[m][1]

                    # 计算标准差
                    A = np.array([L1, L2])
                    var = np.corrcoef(A)
                    s = a[m][0], a[n][0]  # 103699
                    q = a[n][0], a[m][0]  # 699103
                    if s in finallist:
                        finallist.setdefault(s, []).append(var[0][1])
                    elif q in finallist:
                        finallist.setdefault(s, []).append(var[0][1])
                    elif s and q not in finallist:
                        for i in range(currday):
                            finallist.setdefault(s, []).append(np.nan)
                        finallist.setdefault(s, []).append(var[0][1])
                    else:
                        finallist.setdefault(s, []).append(np.nan)

            """
            print(sheet)
            print(a)
            """
            # 清空用于计算每天信息的列表字典，为第二天做准备
            currday += 1
            # 这一天没出现的线路置空
            for k in finallist:
                lens = len(finallist.get(k))
                for i in range(currday - lens):
                    if lens < currday:
                        finallist.setdefault(k, []).append(np.nan)
                    else:
                        break
            sheet.clear()
            sheet.setdefault(line[5], []).append(line[1])
            sheet.setdefault(line[5], []).append(line[4])
            sheet.setdefault(line[5], []).append(1)
            # 该车第一次出现
            sheet.setdefault(line[5], []).append(1)
            # 该车最后一次出现的时间戳
            sheet.setdefault(line[5], []).append(line[1])
            List.clear()
            Lines.clear()
day = line[0]
i = 0
# 统计同一线路刷卡数据，并汇总成二维列表
for key in sheet.keys():
    # 路线刷卡数据不存在，写入List
    if sheet[key][1] not in List:
        Lines.append([])
        Lines[i].append(sheet[key][2])
        List.setdefault(sheet[key][1], i)
        i += 1
        flag = sheet[key][1]
        # 两个相邻的同路线刷卡数据合并
    elif sheet[key][1] in List and flag == sheet[key][1]:
        Lines[List.get(flag)][-1] = Lines[List.get(flag)][-1] + sheet[key][2]
    elif sheet[key][1] in List and flag != sheet[key][1]:
        Lines[List.get(sheet[key][1])].append(sheet[key][2])
        flag = sheet[key][1]
    # 将线路与相应刷卡数序列匹配成列表形式
a = list(zip(List.keys(), Lines))
for m in range(len(a) - 1):
    for n in range(m + 1, len(a)):
        # 删除元素使列表长度相等
        if len(a[m][1]) > len(a[n][1]):
            L1 = a[m][1][0:len(a[n][1])]
            L2 = a[n][1]
        else:
            L2 = a[n][1][0:len(a[m][1])]
            L1 = a[m][1]
            # 计算标准差
            A = np.array([L1, L2])
            var = np.corrcoef(A)
            s = a[m][0], a[n][0]  # 103699
            q = a[n][0], a[m][0]  # 699103
            if s in finallist:
                finallist.setdefault(s, []).append(var[0][1])
            elif q in finallist:
                finallist.setdefault(s, []).append(var[0][1])
            elif s and q not in finallist:
                # 之前没来的 加入 nan表识
                for i in range(currday):
                    finallist.setdefault(s, []).append(np.nan)
                finallist.setdefault(s, []).append(var[0][1])
            else:
                finallist.setdefault(s, []).append(np.nan)

"""
    print(sheet)
    print(a)
    """

# 清空用于计算每天信息的列表字典，为第二天做准备
currday += 1
for k in finallist:
    lens = len(finallist.get(k))
    for i in range(currday - lens):
        if lens < currday:
            finallist.setdefault(k, []).append(np.nan)
        else:
            break
            '''
        
for k in finallist:
    print(finallist.get(k))
    '''
pd.DataFrame(finallist).to_csv(r"C:\Users\lynn\Desktop\92days_correlation.csv")
s = 0
finallistz = {}
finallistf = {}
for k in finallist:
    for i in range(len(finallist.get(k))):
        if np.isnan(finallist.get(k)[i]):
            pass
        else:
            s += float(finallist.get(k)[i])
            print(s)
    if (s >= 0):
        for i in range(len(finallist.get(k))):
            finallistz.setdefault(k,[]).append(finallist.get(k)[i])
    elif(s < 0):
        for i in range(len(finallist.get(k))):
            finallistf.setdefault(k,[]).append(finallist.get(k)[i])
    s = 0
pd.DataFrame(finallistz).to_csv(r"C:\Users\lynn\Desktop\92days_zcorrelation.csv")
pd.DataFrame(finallistf).to_csv(r"C:\Users\lynn\Desktop\92days_fcorrelation.csv")



