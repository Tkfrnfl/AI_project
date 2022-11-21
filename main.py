#-*- coding:utf-8 -*-
import sys, os 
sys.path.append(os.pardir)
import numpy as np 
from PIL import Image 
import pandas as pd
import csv
import math
from NN import NN
#import tensorflow as tf
# csv_data=pd.read_csv('./FIFAallMatchBoxData.csv')

#csv 데이터를 각 나라별 데이터로 재가공
#경기에 긍정적 영향을주는것, 부정적 영향을 주는걸로 2차분류
f=open('./FIFAallMatchBoxData.csv',"r")
csv_data=csv.reader(f)
data_list=[]
for i in csv_data:
    data_list.append(i)

f=open('./international-international-friendlies-matches-2018-to-2018-stats.csv',"r")
csv_data=csv.reader(f)
data_list2=[]
for i in csv_data:
    data_list2.append(i)    
f=open('./international-international-friendlies-matches-2019-to-2019-stats.csv',"r")
csv_data=csv.reader(f)
for i in csv_data:
    data_list2.append(i)     
f=open('./international-international-friendlies-matches-2020-to-2020-stats.csv',"r")
csv_data=csv.reader(f)
for i in csv_data:
    data_list2.append(i)   
f=open('./international-international-friendlies-matches-2021-to-2021-stats.csv',"r")
csv_data=csv.reader(f)
for i in csv_data:
    data_list2.append(i)   

# country_list=np.zeros(shape=(0))
# #country_goal=tf.placeholder(tf.float32,[None])

# # country_result_posi=tf.Variable([None,None],dtype=tf.float32)
# # country_result_nega=tf.Variable([None,None],dtype=tf.float32)

# country_result_posi=np.zeros(shape=(math.ceil(len(data_list)/2),math.ceil(len(data_list)/2)))
# country_result_nega=np.zeros(shape=(math.ceil(len(data_list)/2),math.ceil(len(data_list)/2)))
#country_result_posi=[[0]*math.ceil(len(data_list)/2) for i in range(1) ]
# print(tf.__version__)
#country_result_posi=[0 for _ in range(math.ceil(len(data_list)/2))]
country_result_posi=[]
country_result_nega=[]
country_list=[]

for i,val in enumerate(data_list):          #데이터 가공
    if i !=0:
        if val[1]not in country_list:
            country_list.append(val[1])
        if val[2] not in country_list:
            country_list.append(val[2])
        data_list_h=[]                          #긍정 데이터 가공
        data_list_a=[]
        if val[5] !="0":                        #점유율 데이터가 0 아닌경우만 저장
            data_list_h.append([val[1],val[3],val[5],val[7],val[9]])
            data_list_a.append([val[2],val[4],val[6],val[8],val[10]])
            country_result_posi.append(data_list_h)
            country_result_posi.append(data_list_a)

        data_list_h_ne=[]                          #부정 데이터 가공
        data_list_a_ne=[]
        data_list_h_ne.append([val[1],val[3],val[11],val[13],val[15]])
        data_list_a_ne.append([val[2],val[4],val[12],val[14],val[16]])
        country_result_nega.append(data_list_h_ne)
        country_result_nega.append(data_list_a_ne)


# for j,val_j in enumerate(country_list):
#     for i,val in enumerate(country_result_posi):
#         if val_j in val[0]:
#             print(val[0])

for i,val in enumerate(data_list2):          #데이터 가공
    if i !=0:
        if val[1]not in country_list:
            country_list.append(val[1])
        if val[2] not in country_list:
            country_list.append(val[2])
        data_list_h=[]                          #긍정 데이터 가공
        data_list_a=[]
        if val[3] !="-1" and val[6] !="-1"and val[8] !="-1"and val[13] !="-1":                        #점유율 데이터가 0 아닌경우만 저장
            data_list_h.append([val[1],val[3],val[6],val[8],val[13]])
            data_list_a.append([val[2],val[4],val[7],val[9],val[14]])
            country_result_posi.append(data_list_h)
            country_result_posi.append(data_list_a)

NN(country_result_posi,country_result_nega,"Denmark")




