import folium
from folium import plugins
import os
import csv
import numpy as np
import math
import pandas as pd
import webbrowser



m = folium.Map(location=[24.503610, 118.120873],zoom_start=12) #绘制地图：中心坐标和放大倍数
marker_cluster = plugins.MarkerCluster().add_to(m) #点聚合
place = [] #经纬度列表
sheet = [] #站点名称列表
path = "C:/Users/lynn/Desktop/GPS轨迹"  #文件夹位置
#打开文件夹，遍历文件夹中的所有文件
files= os.listdir(path) 
for file in files:
    with open(os.path.join(path,file),'r') as f:
        data = csv.reader(f)
        for line in data:
            if float(line[4]) != 0 and float(line[3]) != 0: #当经纬度均不为零时
                place.append([float(line[4])/1000000,float(line[3])/1000000])  #存储在经纬度列表中
            if line[6] not in sheet and line[6] != "Driving": #当文件中的第7列没有在站点名称列表中出现或者不是“Driving”状态时
                sheet.append(line[6]) #存储在站点名称列表中
                #对每个站点画圈，标出站点名称和相关度（未完成）
                text = folium.Html(
                    '<b>站点名称: {}</b></br><b>正相关度: {}</b></br> <b>负相关度: {}</b></br>'.format(
                        line[6],
                        line[7],
                        line[8]),
                    script=True
                )
                folium.CircleMarker(
                    location=[float(line[4])/1000000,float(line[3])/1000000],
                    radius = 20 ,
                    opacity = 0.8,
                    color='orange', #圆圈颜色
                    weight=float(line[7])/(float(line[7])+abs(float(line[8])))*20,
                    fill_opacity=0.8, #填充透明度
                    fill_color='#9933CC', #填充颜色
                    popup = folium.Popup(text,max_width=2650)
                ).add_to(marker_cluster)

        folium.PolyLine(place,color = 'blue',opacity=0.6).add_to(m) #对一条线路的每个GPS数据进行连线
        place.clear() #清空这条线路的经纬度数据，为存储下一条线路经纬度数据做准备

m.save('map.html') #存储为html文件
webbrowser.open('map.html') #在浏览器中打开
