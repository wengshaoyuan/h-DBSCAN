# -*- coding: utf-8 -*-
import pandas as pd
df =  pd.read_csv("station.csv", encoding='gbk')

data=df[['STATION_NAME','LINE_NAME','CARTYPE']]

finalData=data[data['CARTYPE']==0]


#输入要计算的线路的号码
LineNmae1='109'
LineNmae2='101'

#计算线路的站点数
finalDataLine1=finalData[finalData['LINE_NAME']==LineNmae1]
finalDataLine2=finalData[finalData['LINE_NAME']==LineNmae2]

#提取不同线路的站点名字
finalDataLine1Station=set(finalDataLine1['STATION_NAME'])
finalDataLine2Station=set(finalDataLine2['STATION_NAME'])

len1=len(finalDataLine1Station)
len2=len(finalDataLine2Station)


#相同站点的个数
sameLen=len(finalDataLine1Station & finalDataLine2Station)
#相同站点的名称
sameNmae=finalDataLine1Station & finalDataLine2Station



#计算王弟鑫方法的合作系数
cooP=((len1-sameLen)*(len2-sameLen))/((len1-1)*(len2-1))
print(cooP)




